//! One process's Memory over host pages, standalone: passes, records, retained ranges, the
//! current step's classes and free-only work.
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

fn open(case: &str, memory: u64) -> (Scope, Memory, GpuId) {
    let g = GpuId { uuid: format!("m{}-{case}", std::process::id()), model: "host".into(), memory };
    let scope = Scope::new(&g.uuid);
    let mem = Memory::open(g.clone(), Box::new(HostBackend::new(memory))).unwrap();
    (scope, mem, g)
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
    let (_scope, mem, g) = open("late", 16 * MB);
    let id = mem.load(set(dir.path(), "l", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/l".into()], passes: 2 }] }).unwrap();
    let first = r.pass(id, "k").unwrap();
    let second = r.pass(id, "k").unwrap();
    drop(first);
    drop(mem.bind(id, 0, 0).unwrap());
    assert_eq!(r.stats().trace.len(), 1, "the bind is still the request's");
    drop(second);
}

/// The vertical slice: a cold request's record carries the fills of the loads before it, by
/// source, so its cold cost shows; a warm one after it carries none.
#[test]
fn a_cold_requests_record_carries_its_loads() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, mem, g) = open("loadstats", 16 * MB);
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

#[test]
fn retained_addresses_survive_unload_and_refill_without_pinning_pages() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, mem, g) = open("retained-refill", 8 * MB);
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
    let (_scope, mem, g) = open("retained-identity", 8 * MB);
    let id = mem.load(set(dir.path(), "identity", 1, &g.uuid), Level::Ram).unwrap();
    let binding = mem.retain(id, &[0]).unwrap();
    mem.fill(id, Fill::Declared("changed-bake".into()));
    assert!(matches!(binding.prepare(0), Err(varena::Error::StaleBinding(_))));
}

#[test]
fn a_range_larger_than_capacity_unwinds_all_preparation_claims() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, mem, g) = open("retained-exhausted", 2 * REGION_ALIGN);
    let id = mem.load(set(dir.path(), "large-range", 3, &g.uuid), Level::Ram).unwrap();
    let all = mem.retain(id, &[0, 1, 2]).unwrap();
    assert!(matches!(all.prepare(0), Err(varena::Error::Exhausted(_))));
    let one = mem.retain(id, &[0]).unwrap();
    assert_eq!(one.prepare(0).unwrap().len(), 1, "failed preparation strands no physical claims");
}

/// Lead's ruling (S4, SDXL's tiled VAE decode, one pass per tile): a set in the current step is
/// W until the plan's cursor leaves the step, however many passes it runs; its later tiles find
/// its pages in use, never N.
#[test]
fn a_tiled_step_keeps_its_pages_w_through_every_tile() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, mem, g) = open("tiles", 16 * MB);
    let id = mem.load(set(dir.path(), "vae", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("tiles", plan("vae")).unwrap();
    let class = |c: varena::table::Class| -> u64 { mem.classes()[c.index()] };
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
/// free-only bind, and an idle request's binds, are `Exhausted` at once when free bytes lack,
/// taking nothing; an ordinary bind then takes the idle pages.
#[test]
fn idle_work_takes_free_bytes_only() {
    let dir = tempfile::tempdir().unwrap();
    let (_scope, mem, g) = open("idle-work", 8 * MB);
    mem.load(set(dir.path(), "n", 3, &g.uuid), Level::Gpu).unwrap();
    let a = mem.load(set(dir.path(), "a", 2, &g.uuid), Level::Ram).unwrap();
    let mapped = || mem.classes().iter().sum::<u64>();
    let first = mem.bind_free(a, 0, 0).unwrap();
    assert_eq!(mapped(), 4 * 2 * MB);
    assert!(matches!(mem.bind_free(a, 1, 0), Err(varena::Error::Exhausted(_))), "no free page left");
    let r = mem.request_idle("capture", plan("a")).unwrap();
    let pass = r.pass(a, "k").unwrap();
    assert!(matches!(mem.bind(a, 1, 0), Err(varena::Error::Exhausted(_))), "an idle request's bind takes nothing");
    assert_eq!(mapped(), 4 * 2 * MB, "nothing was taken");
    pass.end(0);
    drop((r, first));
    let r = mem.request("ordinary", plan("a")).unwrap();
    let pass = r.pass(a, "k").unwrap();
    drop(mem.bind(a, 1, 0).unwrap());
    assert!(r.stats().evicted[varena::table::Class::N.index()] >= 2 * MB, "an ordinary bind takes an idle page: {:?}", r.stats());
    pass.end(0);
}
