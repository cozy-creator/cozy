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
        pass.end(Measured { allocated_peak: peak, slack: 0, sites: vec![("ffn".into(), peak / 2)] }, Torch::default());
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
    assert!(matches!(r.oom(Some(MB), site, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut("ffn"), 1);
    assert!(matches!(r.oom(Some(MB), site, Torch::default()), Next::Cut(2)));
    // Review 1, finding 4: a collective site never cuts mid-pass. It recovers locally, its
    // finer level is proposed, and it takes effect when the ranks agree at the next pass.
    let collective = Some(Site { name: "allgather", collective: true });
    let next = r.oom(Some(MB), collective, Torch::default());
    assert!(matches!(next, Next::Park(_) | Next::Exhausted(_)), "{next:?}");
    assert_eq!(r.cut("allgather"), 0, "unchanged within the pass");
    drop(pins);
    drop(pass);
    let next = r.pass(id, "1x8", Torch::default()).unwrap();
    assert_eq!(next.levels(), vec![("allgather".to_string(), 1)], "proposed for the agreement");
    assert_eq!(r.cut("allgather"), 0);
    next.agree(vec![("allgather".into(), 2)]);
    assert_eq!(r.cut("allgather"), 2, "the ranks' maximum");
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

/// Review 1, finding 3: a Memory that served custody, once dropped, leaves no row publishing its
/// bytes: custody's row is its own and holds only orphans.
#[test]
fn a_dropped_custody_holder_leaves_no_zombie_row() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("m{}-zombie", std::process::id()), model: "host".into(), memory: 16 * MB };
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    mem.load(set(dir.path(), "z", 2, &g.uuid), Level::Gpu).unwrap();
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let held = |l: &varena::Ledger| l.status().rows.iter().map(|r| r.classes.iter().sum::<u64>()).sum::<u64>();
    assert_eq!(held(&ledger), 4 * MB);
    drop(mem);
    // Its kept pages are orphans now (class O on custody's row); nothing else is published.
    let rows = ledger.status().rows;
    assert!(rows.iter().all(|r| r.classes.iter().enumerate().all(|(c, b)| c == 1 || *b == 0)), "{rows:?}");
    assert_eq!(rows.iter().map(|r| r.classes[1]).sum::<u64>(), 4 * MB, "S5: class O shows the orphans");
}

/// S5: `fits(package, "")` means the package's widest measured shape.
#[test]
fn an_empty_shape_fits_by_the_widest_measured() {
    let (ledger, mem, _g) = open("widest", 16 * MB);
    ledger.restore(&[varena::Peak { key: "pkg|1x1".into(), bytes: 2 * MB, sites: vec![] }, varena::Peak { key: "pkg|4x4".into(), bytes: 17 * MB, sites: vec![] }]).unwrap();
    let busy = mem.request("busy", Plan::default()).unwrap();
    assert!(ledger.fits("pkg", "1x1"));
    assert!(!ledger.fits("pkg", ""), "the widest (17 MB) beside a running request does not fit");
    drop(busy);
    assert!(ledger.fits("pkg", ""), "an idle GPU fits anything");
}
