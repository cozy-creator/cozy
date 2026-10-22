//! One process's Memory over host pages: rooms bounded by measured shapes, the cut memo, and
//! the OOM answers for cut and collective sites.
// Its sets load from TensorFS sources, which are Unix's so far (`tensorfs::io::source`).
#![cfg(unix)]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Plan, SetSpec, Step};

const MB: u64 = 1 << 20;

fn open(case: &str, memory: u64) -> (Scope, varena::Ledger, Memory, GpuId) {
    let g = GpuId { uuid: format!("m{}-{case}", std::process::id()), model: "host".into(), memory };
    let scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(memory))).unwrap();
    (scope, ledger, mem, g)
}

/// This binary's namespace for tables and service names (never a machine's), and the test's
/// table removed when it ends.
// Fixtures are isolated from one another; concurrency inside a case remains unchanged.
static TEST_SCOPE: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());
struct Scope {
    uuid: String,
    _serial: parking_lot::ReentrantMutexGuard<'static, ()>,
}

impl Scope {
    fn new(uuid: &str) -> Scope {
        let serial = TEST_SCOPE.lock();
        assert!(varena::os::namespace(&format!("test{}-", std::process::id())));
        Scope { uuid: uuid.to_string(), _serial: serial }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        varena::os::remove_table(&self.uuid);
    }
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

/// Review 1, finding 11: dropping an earlier pass after the next one started keeps the next
/// one's state (binds stay its request's).
#[test]
fn a_late_dropped_pass_leaves_the_next_one() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("late", 16 * MB);
    let id = mem.load(set(dir.path(), "l", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/l".into()], passes: 2 }] }).unwrap();
    let first = r.pass(id, "k").unwrap();
    let second = r.pass(id, "k").unwrap();
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
    let _scope = Scope::new(&g.uuid);
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    mem.load(set(dir.path(), "z", 2, &g.uuid), Level::Gpu).unwrap();
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let held = |l: &varena::Ledger| l.status().rows.iter().map(|r| r.classes.iter().enumerate().filter(|(c, _)| *c != 1).map(|(_, b)| b).sum::<u64>()).sum::<u64>();
    assert_eq!(held(&ledger), 4 * MB);
    drop(mem);
    // Final public drop transfers ownership to retirement; the row remains charged until
    // completed device work and custody handoff are finished on that independent worker.
    while held(&ledger) != 0 { ledger.await_event().unwrap(); }
    assert_eq!(held(&ledger), 0, "{:?}", ledger.status().rows);
}

/// S5: after its holder ends, its kept pages show as class O on custody's row.
#[test]
fn orphans_show_as_class_o() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, ledger, mem, g) = open("orphans", 16 * MB);
    mem.load(set(dir.path(), "o", 2, &g.uuid), Level::Gpu).unwrap();
    drop(mem);
    // The custody service sees the connection end on its own thread; it rings the ledger.
    while ledger.status().rows.iter().map(|r| r.classes[1]).sum::<u64>() < 4 * MB {
        ledger.await_event().unwrap();
    }
}

/// S5 review, 1: a restarted TensorD drops the dispatches it finds (a dead TensorD's), so the
/// GPU never reads as busy forever.
#[test]
fn a_restarted_tensord_drops_stale_dispatches() {
    let g = GpuId { uuid: format!("m{}-dispatch", std::process::id()), model: "host".into(), memory: 16 * MB };
    let _scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    ledger.dispatch("r1", "pkg", "1x1").unwrap();
    assert_eq!(ledger.status().dispatched.len(), 1);
    drop(ledger);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    assert!(ledger.status().dispatched.is_empty());
}


/// The vertical slice: a cold request's record carries the fills of the loads before it, by
/// source, so its cold cost shows; a warm one after it carries none.
#[test]
fn a_cold_requests_record_carries_its_loads() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("loadstats", 16 * MB);
    let id = mem.load(set(dir.path(), "ls", 2, &g.uuid), Level::Gpu).unwrap();
    let cold = mem.request("cold", plan("ls")).unwrap();
    let s = cold.stats();
    assert_eq!(s.moved.store.bytes, 2 * REGION_ALIGN, "the load's fills, from the store");
    assert_eq!(s.trace.iter().filter(|r| r.origin == varena::Origin::Store).count(), 2, "{:?}", s.trace);
    let pass = cold.pass(id, "k").unwrap();
    drop(mem.bind(id, 0, 0).unwrap());
    drop(pass);
    let s = cold.end();
    assert_eq!(s.moved.store.bytes, 2 * REGION_ALIGN);
    assert_eq!(s.trace.len(), 3, "two load fills and one bind: {:?}", s.trace);
    let warm = mem.request("warm", plan("ls")).unwrap();
    assert_eq!(warm.stats().moved.store.bytes, 0, "nothing filled since");
}

/// v8 (a GPU test's 1 GiB limit capped every executor on the pod): TensorD sets the table's
/// capacity exactly, whatever an earlier opener chose, and its limit never outlives it; a
/// standalone opener's limit caps its own row only.
#[test]
fn only_tensord_sets_the_capacity_and_an_openers_limit_is_its_rows() {
    let g = GpuId { uuid: format!("m{}-limit", std::process::id()), model: "host".into(), memory: 64 * MB };
    let _scope = Scope::new(&g.uuid);
    let capacity = |l: &varena::Ledger| l.status().capacity;
    // A standalone opener with a small limit, gone before TensorD starts: nothing of it stays.
    let small = Memory::open(g.clone(), Some(varena::Grant::open(&g, Some(8 * MB)).unwrap()), Box::new(HostBackend::new(g.memory))).unwrap();
    assert_eq!(small.free(), 8 * MB, "its own row is capped");
    drop(small);
    let ledger = varena::Ledger::create(&g, Some(12 * MB)).unwrap();
    assert_eq!(capacity(&ledger), 12 * MB, "TensorD's limit, exactly");
    drop(ledger);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    assert_eq!(capacity(&ledger), 64 * MB, "the limit lifted: the card's");
    let executor = Memory::open(g.clone(), Some(ledger.grant(std::process::id()).unwrap()), Box::new(HostBackend::new(g.memory))).unwrap();
    assert_eq!(executor.free(), 64 * MB, "an executor gets the whole card");
    drop(executor);
    // (A live joiner's limit beside a peer: processes::a_joiners_limit_leaves_its_peers_room.)
}

/// S5 r3: a load records its set's bytes (its layout's span) on its GPU's table, persisted as
/// Peak::Set; placement homes a set that does not fit beside GPU 0's pinned pages on the GPU
/// with room, and keeps it on GPU 0 once they let go.
#[test]
fn a_set_goes_where_it_fits_and_stays_on_gpu_0_when_it_can() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, ledger, mem, g) = open("place0", 8 * MB);
    let other = GpuId { uuid: format!("m{}-place1", std::process::id()), model: "host".into(), memory: 8 * MB };
    let _other_scope = Scope::new(&other.uuid);
    let second = varena::Ledger::create(&other, None).unwrap();
    let big = mem.load(set(dir.path(), "big", 3, &g.uuid), Level::Ram).unwrap();
    mem.load(set(dir.path(), "te", 2, &g.uuid), Level::Ram).unwrap();
    assert!(ledger.peaks().contains(&varena::Peak::Set { key: "pkg@1/te".into(), bytes: 2 * REGION_ALIGN }));
    let r = mem.request("r", plan("big")).unwrap();
    let pass = r.pass(big, "k").unwrap();
    let pins: Vec<_> = (0..3).map(|b| mem.bind(big, b, 0).unwrap()).collect();
    let te = [("pkg@1/te", None)];
    assert_eq!(varena::Ledger::place(&[&ledger, &second], "pkg", "", None, &te).unwrap(), vec![1], "GPU 0 holds 2 MB beside the running pass");
    drop(pins);
    pass.end(0);
    r.end();
    assert_eq!(varena::Ledger::place(&[&ledger, &second], "pkg", "", None, &te).unwrap(), vec![0], "idle, GPU 0 keeps it");
    assert_eq!(varena::Ledger::place(&[&ledger, &second], "pkg", "", None, &[("pkg@1/te", Some(6 * MB))]).unwrap(), vec![0], "a layout's bytes, given");
}

/// Review 4, L1: placement keeps only the package's compute room free on GPU 0, so measuring a
/// run never moves a set the next run: place, run, place again, three times over.
#[test]
fn placement_is_stable_across_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, ledger, mem, g) = open("stable0", 16 * MB);
    let other = GpuId { uuid: format!("m{}-stable1", std::process::id()), model: "host".into(), memory: 16 * MB };
    let _other_scope = Scope::new(&other.uuid);
    let second = varena::Ledger::create(&other, None).unwrap();
    let unet = mem.load(set(dir.path(), "unet", 2, &g.uuid), Level::Ram).unwrap();
    let te = mem.load(set(dir.path(), "te", 3, &g.uuid), Level::Ram).unwrap();
    let mut homes = vec![];
    for run in 0..3 {
        let home = varena::Ledger::place(&[&ledger, &second], "pkg", "1x8", None, &[("pkg@1/te", None)]).unwrap();
        homes.push(home.clone());
        // The run: te's pages on GPU 0 only when placed there; a 4 MB room beside them.
        let id = format!("run{run}");
        ledger.dispatch(&id, "pkg", "1x8").unwrap();
        let local = home == vec![0];
        let sets: Vec<String> = ["unet", "te"].iter().take(if local { 2 } else { 1 }).map(|s| format!("pkg@1/{s}")).collect();
        let r = mem.request(&id, Plan { steps: vec![Step { sets, passes: 1 }] }).unwrap();
        for (set, blocks) in [(unet, 2), (te, 3)].into_iter().take(if local { 2 } else { 1 }) {
            let pass = r.pass(set, "k").unwrap();
            let pins: Vec<_> = (0..blocks).map(|b| mem.bind(set, b, 0).unwrap()).collect();
            drop(pins);
            pass.end(4 * MB);
        }
        r.end();
    }
    assert_eq!(homes, vec![vec![0]; 3], "te stays on GPU 0, measured or not");
}

/// S3 gpu10: a need beyond what this row could ever have (the budget less its context, the
/// hook's segments and its pinned pages) is Exhausted at once: nothing is taken, and the pass's
/// next bind still finds its page.
#[test]
fn an_impossible_need_is_exhausted_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("impossible", 8 * MB);
    let id = mem.load(set(dir.path(), "i", 2, &g.uuid), Level::Gpu).unwrap();
    let r = mem.request("r", plan("i")).unwrap();
    let pass = r.pass(id, "k").unwrap();
    drop(mem.bind(id, 0, 0).unwrap());
    assert!(matches!(mem.make_room(64 * MB), Err(varena::Error::Exhausted(_))));
    assert!(r.stats().evicted.iter().all(|e| *e == 0), "{:?}", r.stats());
    drop(mem.bind(id, 1, 0).unwrap());
    assert_eq!(r.stats().trace.last().map(|t| t.origin), Some(varena::Origin::Hit));
    pass.end(0);
}

/// Host-OOM, the varena side: a request records its anonymous host peak (Peak::Host; file
/// mappings aside). While it runs unmeasured its process is an unmeasured call, and once
/// measured the call carries that peak. Linux: the peak is read from `/proc` (elsewhere a request
/// is unmeasured).
#[cfg(target_os = "linux")]
#[test]
fn a_request_records_its_anonymous_host_peak() {
    let (_scope, ledger, mem, _g) = open("host", 8 * MB);
    ledger.dispatch("first", "pkg", "1x1").unwrap();
    let r = mem.request("first", Plan { steps: vec![] }).unwrap();
    let unmeasured = |calls: Option<Vec<varena::HostCall>>| calls.is_some_and(|c| c.len() == 1 && c[0].pid == std::process::id() && c[0].peak.is_none());
    assert!(unmeasured(ledger.host_calls()), "a running request with no host peak");
    let held: Vec<u8> = vec![1; (64 * MB) as usize];
    let stats = r.end();
    assert!(stats.host.is_some_and(|h| h >= 64 * MB), "{:?}", stats.host);
    let recorded = ledger.peaks().into_iter().find_map(|p| match p {
        varena::Peak::Host { key, bytes } if key == "pkg|1x1" => Some(bytes),
        _ => None,
    });
    assert!(recorded.is_some_and(|b| b >= 64 * MB), "{recorded:?}");
    ledger.dispatch("second", "pkg", "1x1").unwrap();
    let r = mem.request("second", Plan { steps: vec![] }).unwrap();
    let calls = ledger.host_calls().unwrap();
    assert!(calls.len() == 1 && calls[0].peak == recorded, "measured now: {calls:?}");
    drop(held);
    r.end();
}

#[test]
fn retained_addresses_survive_unload_and_refill_without_pinning_pages() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("retained-refill", 8 * MB);
    let id = mem.load(set(dir.path(), "retained", 2, &g.uuid), Level::Ram).unwrap();
    let binding = mem.retain(id, &[1, 0, 1]).unwrap();
    let first = binding.prepare(0).unwrap();
    let addresses: Vec<_> = first.iter().map(|p| p.ptr).collect();
    assert_eq!(addresses.len(), 2, "ranges are deduplicated and ordered");
    assert!(first.iter().all(|p| unsafe { *(p.ptr as *const u8) } == 3));
    drop(first);
    mem.unload(id);
    mem.settle();
    let second = binding.prepare(0).unwrap();
    assert_eq!(second.iter().map(|p| p.ptr).collect::<Vec<_>>(), addresses);
    assert!(second.iter().all(|p| unsafe { *(p.ptr as *const u8) } == 3));
    drop(second);
    drop(binding);
    mem.settle();
    assert!(mem.bind(id, 0, 0).is_err(), "final address owner retires the unloaded set");
}

#[test]
fn retained_identity_change_refuses_preparation_before_use() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("retained-identity", 8 * MB);
    let id = mem.load(set(dir.path(), "identity", 1, &g.uuid), Level::Ram).unwrap();
    let binding = mem.retain(id, &[0]).unwrap();
    mem.fill(id, Fill::Declared("changed-bake".into()));
    assert!(matches!(binding.prepare(0), Err(varena::Error::StaleBinding(_))));
}

#[test]
fn a_range_larger_than_capacity_unwinds_all_preparation_claims() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, _l, mem, g) = open("retained-exhausted", 2 * REGION_ALIGN);
    let id = mem.load(set(dir.path(), "large-range", 3, &g.uuid), Level::Ram).unwrap();
    let all = mem.retain(id, &[0, 1, 2]).unwrap();
    assert!(matches!(all.prepare(0), Err(varena::Error::Exhausted(_))));
    let one = mem.retain(id, &[0]).unwrap();
    assert_eq!(one.prepare(0).unwrap().len(), 1, "failed preparation strands no physical claims");
}

#[test]
fn a_grant_keeps_its_table_generation_when_the_ledger_is_dropped() {
    let g = GpuId { uuid: format!("m{}-grant-generation", std::process::id()), model: "host".into(), memory: 64 * MB };
    let _scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, Some(12 * MB)).unwrap();
    let grant = varena::Grant::open(&g, None).unwrap();
    drop(ledger);
    drop(varena::Grant::open(&g, Some(8 * MB)).unwrap());
    let memory = Memory::open(g.clone(), Some(grant), Box::new(HostBackend::new(g.memory))).unwrap();
    // The table outlived its TensorD (whose limit went with it); the other opener's cap was its own.
    assert_eq!(memory.free(), 64 * MB, "the grant opens the same live generation");
}

/// Lead's ruling (S4, SDXL's tiled VAE decode, one pass per tile): a set in the current step is
/// W until the plan's cursor leaves the step, however many passes it runs; its later tiles find
/// its pages in use, never N.
#[test]
fn a_tiled_step_keeps_its_pages_w_through_every_tile() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, ledger, mem, g) = open("tiles", 16 * MB);
    let id = mem.load(set(dir.path(), "vae", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("tiles", plan("vae")).unwrap();
    let class = |c: varena::table::Class| -> u64 { ledger.status().rows.iter().map(|r| r.classes[c.index()]).sum() };
    for tile in 0..3 {
        let pass = r.pass(id, "k").unwrap();
        for b in 0..2 {
            drop(mem.bind(id, b, 0).unwrap());
        }
        pass.end(0);
        assert_eq!((class(varena::table::Class::W), class(varena::table::Class::N)), (2 * REGION_ALIGN, 0), "tile {tile} of a step declared once");
    }
    drop(r);
}



/// Idle work (S4's captures off the request path, S3's free-only hold) takes free bytes only: a
/// free-only bind, and an idle request's binds and hook, are `Exhausted` at once when free
/// bytes lack, taking nothing; an ordinary bind then takes the idle pages.
#[test]
fn idle_work_takes_free_bytes_only() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, ledger, mem, g) = open("idle-work", 8 * MB);
    mem.load(set(dir.path(), "n", 3, &g.uuid), Level::Gpu).unwrap();
    let a = mem.load(set(dir.path(), "a", 2, &g.uuid), Level::Ram).unwrap();
    // Mapped pages, whatever their class (a Keep's protect pins one a moment).
    let mapped = || ledger.status().rows.iter().map(|r| r.classes.iter().skip(varena::table::Class::S.index()).sum::<u64>()).sum::<u64>();
    let first = mem.bind_free(a, 0, 0).unwrap();
    assert_eq!(mapped(), 4 * 2 * MB);
    assert!(matches!(mem.bind_free(a, 1, 0), Err(varena::Error::Exhausted(_))), "no free page left");
    let r = mem.request_idle("capture", plan("a")).unwrap();
    let pass = r.pass(a, "k").unwrap();
    assert!(matches!(mem.make_room(2 * MB), Err(varena::Error::Exhausted(_))), "an idle request's hook takes nothing");
    assert_eq!(mapped(), 4 * 2 * MB, "nothing was taken");
    pass.end(0);
    drop((r, first));
    let r = mem.request("ordinary", plan("a")).unwrap();
    let pass = r.pass(a, "k").unwrap();
    drop(mem.bind(a, 1, 0).unwrap());
    assert!(r.stats().evicted[varena::table::Class::N.index()] >= 2 * MB, "an ordinary bind takes an idle page: {:?}", r.stats());
    pass.end(0);
}
