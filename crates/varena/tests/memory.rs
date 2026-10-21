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
    let pass = r.pass(id, shape, Torch::default()).unwrap();
    let w = r.stats().peak;
    if let Some(peak) = measured {
        pass.end(Measured { allocated_peak: peak, slack: 0, sites: vec![varena::SitePeak { name: "ffn".into(), peak: peak / 2, collective: false }] }, Torch::default());
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
    let pass = r.pass(id, "1x8", Torch::default()).unwrap();
    let pins: Vec<_> = (0..2).map(|b| mem.bind(id, b, 0).unwrap()).collect();
    let site = Some(Site { name: "ffn", collective: false });
    // The free bytes went to the pages; every page is pinned: the op cuts, level by level.
    assert!(!pass.measured() && !r.measured(), "no room measured at this shape yet");
    assert!(matches!(r.oom(id, Some(MB), site, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut(id, "ffn"), 1);
    assert!(matches!(r.oom(id, Some(MB), site, Torch::default()), Next::Cut(2)));
    // Review 1, finding 4: a collective site never cuts mid-pass. It recovers locally, its
    // finer level is proposed, and it takes effect when the ranks agree at the next pass.
    let collective = Some(Site { name: "allgather", collective: true });
    let next = r.oom(id, Some(MB), collective, Torch::default());
    assert!(matches!(next, Next::Park(_) | Next::Exhausted(_)), "{next:?}");
    assert_eq!(r.cut(id, "allgather"), 0, "unchanged within the pass");
    drop(pins);
    drop(pass);
    let next = r.pass(id, "1x8", Torch::default()).unwrap();
    assert_eq!(next.levels(), vec![("allgather".to_string(), 1)], "proposed for the agreement");
    assert_eq!(r.cut(id, "allgather"), 0);
    next.agree(vec![("allgather".into(), 2)]);
    assert_eq!(r.cut(id, "allgather"), 2, "the ranks' maximum");
}

/// Review 1, finding 11: dropping an earlier pass after the next one started keeps the next
/// one's state (binds stay its request's).
#[test]
fn a_late_dropped_pass_leaves_the_next_one() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("late", 16 * MB);
    let id = mem.load(set(dir.path(), "l", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/l".into()], passes: 2 }] }).unwrap();
    let first = r.pass(id, "1x8", Torch::default()).unwrap();
    let second = r.pass(id, "1x8", Torch::default()).unwrap();
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
    let pa = r.pass(a, "1x8", Torch::default()).unwrap();
    let pb = r.pass(b, "1x8", Torch::default()).unwrap();
    let pins = [mem.bind(a, 0, 0).unwrap(), mem.bind(b, 0, 0).unwrap()];
    let site = Some(Site { name: "ffn", collective: false });
    assert!(matches!(r.oom(a, Some(MB), site, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut(a, "ffn"), 1, "a's pass cut");
    assert_eq!(r.cut(b, "ffn"), 0, "b's pass, started last, untouched");
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
        let p = first.pass(id, "1x8", torch).unwrap();
        p.end(Measured { allocated_peak: latents + grows, slack: 0, sites: vec![] }, torch);
    }
    let peak = first.end().peak;
    assert_eq!(peak, 2 * page + latents + 4 * MB, "the first request's peak is b's pages and its absolute room");
    let second = mem.request("second", plan()).unwrap();
    let remaining = |l: &varena::Ledger| l.status().rows.iter().map(|r| r.remaining).sum::<u64>();
    assert_eq!(remaining(&ledger), 2 * page + latents + 4 * MB, "published absolute: pages, start allocation, growth");
    let p = second.pass(a, "1x8", torch).unwrap();
    assert!(p.measured() && second.measured(), "a's room at 1x8 was measured");
    // Speculation fetched ahead only into what leaves b's absolute room beyond a's free.
    let free = ledger.status().free;
    assert!(free >= 4 * MB, "b's extra room stays free: {free}");
    p.end(Measured { allocated_peak: latents, slack: 0, sites: vec![] }, torch);
    for id in [b, a] {
        let p = second.pass(id, "1x8", torch).unwrap();
        (0..if id == b { 2 } else { 1 }).for_each(|block| drop(mem.bind(id, block, 0).unwrap()));
        p.end(Measured { allocated_peak: latents, slack: 0, sites: vec![] }, torch);
    }
    let stats = second.end();
    assert_eq!(stats.evicted[varena::table::Class::D.index()], 0, "{stats:?}");
}
