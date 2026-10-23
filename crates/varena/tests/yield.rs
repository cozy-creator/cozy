//! VRAM yield (design vram-yield.md): foreign use on a shared GPU gives cozy's bytes back. TensorD's
//! reading (`Ledger::set_foreign`) asks the rows for the deficit and H, idle rows' pages first,
//! then running sets stream; a new foreign process takes idle pages only. Growth waits for a pass
//! boundary and for 2H free. At the floor, a refusal names the foreign process. On Windows the
//! process's budget does the same from its own thread (a fake budget here).
#![cfg(unix)]

use std::fs::File;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Budget, Event, Fill, Fingerprint, GpuId, HostBackend, Ledger, Level, Memory, Physical, Plan, SetSpec, Step, Va};

const P: u64 = REGION_ALIGN;
/// A card the headroom H is 1 GiB of; only the pages made take host memory.
const CARD: u64 = 16 << 30;

struct Scope(String);

impl Scope {
    fn new(case: &str) -> (Scope, GpuId) {
        assert!(varena::os::namespace(&format!("yield{}-", std::process::id())));
        let uuid = format!("yield-{case}-{}", std::process::id());
        (Scope(uuid.clone()), GpuId { uuid, model: "host".into(), memory: CARD })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        varena::os::remove_table(&self.0);
    }
}

fn set(dir: &std::path::Path, name: &str, blocks: u64, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![5u8; (blocks * P) as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: blocks * P };
    let regions = (0..blocks)
        .map(|i| Region { offset: i * P, nbytes: P, span: P, items: vec![Item { offset: i * P, len: P, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: i * P, len: P }), part_end: true }] })
        .collect();
    let layout = Layout {
        parts: (0..blocks).map(|i| Part { what: format!("t{i}"), region: i as u32, offset: i * P, nbytes: P }).collect(),
        regions,
        nbytes: blocks * P,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..blocks).map(|i| vec![format!("t{i}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

fn plan(set: &str) -> Plan {
    Plan { steps: vec![Step { sets: vec![set.into()], passes: 1 }] }
}

/// This process's rows (its Memory's and custody's): what they hold, as the ledger counts it,
/// and their mapped page bytes.
fn row(ledger: &Ledger) -> (u64, u64) {
    ledger.status().rows.iter().filter(|r| r.pid == std::process::id() && r.live).fold((0, 0), |(held, pages), r| {
        let p: u64 = r.classes.iter().enumerate().filter(|(c, _)| *c != 1).map(|(_, b)| b).sum();
        (held + r.context + r.torch + p, pages + p)
    })
}

/// TensorD's reading of a device with `free` bytes free: the bytes no row accounts for, as NVML
/// shows them (card − rows' held − free).
fn reading(ledger: &Ledger, free: u64) -> u64 {
    CARD - row(ledger).0 - free
}

/// A reading with 2H free asks nothing; one with less than H free asks for 2H − free, which takes
/// every idle page.
#[test]
fn a_foreign_reading_takes_the_deficit_and_h_from_idle_pages() {
    let (_scope, g) = Scope::new("deficit");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "idle", 4, &g.uuid), Level::Gpu).unwrap();
    assert_eq!(row(&ledger).1, 4 * P, "the idle set is resident");
    let h = varena::headroom(CARD);
    assert_eq!(ledger.set_foreign(reading(&ledger, 2 * h), false).unwrap(), 0, "2H free: nothing asked");
    assert_eq!(row(&ledger).1, 4 * P);
    let freed = ledger.set_foreign(reading(&ledger, h / 2), false).unwrap();
    assert_eq!((freed, row(&ledger).1), (4 * P, 0), "(freed, pages left): H/2 free asks 3H/2, every idle page");
    drop(mem);
}

/// A new foreign process (`fresh`) takes every idle row's idle pages, with no deficit; a running
/// set is never asked for it.
#[test]
fn a_new_foreign_process_takes_idle_pages_only() {
    let (_scope, g) = Scope::new("fresh");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let idle = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    idle.load(set(dir.path(), "idle", 3, &g.uuid), Level::Gpu).unwrap();
    let room = 2 * varena::headroom(CARD);
    let freed = ledger.set_foreign(reading(&ledger, room), true).unwrap();
    assert_eq!((freed, row(&ledger).1), (3 * P, 0), "(freed, pages left) for a new foreign process");
    // A running row: its working set stays (the rule takes idle pages only).
    let s = idle.load(set(dir.path(), "run", 3, &g.uuid), Level::Ram).unwrap();
    let r = idle.request("r", plan("run")).unwrap();
    let pass = r.pass(s, "k").unwrap();
    let pins: Vec<_> = (0..3).map(|b| idle.bind(s, b, 0).unwrap()).collect();
    drop(pins);
    let before = row(&ledger).1;
    assert_eq!(ledger.set_foreign(reading(&ledger, room), true).unwrap(), 0, "a running row's working set was asked for a new process");
    assert_eq!(row(&ledger).1, before);
    pass.end(0);
    drop(r);
}

/// Short with no idle pages anywhere, the running set gives its upper blocks (its watermark
/// lowers) and the run goes on: every block still binds with its own bytes, and nothing ends it.
#[test]
fn foreign_pressure_streams_the_running_set_and_never_ends_it() {
    let (_scope, g) = Scope::new("stream");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap());
    let blocks = 8;
    let s = mem.load(set(dir.path(), "run", blocks, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("run")).unwrap();
    let pass = r.pass(s, "k").unwrap();
    for b in 0..blocks as u32 {
        drop(mem.bind(s, b, 0).unwrap());
    }
    assert_eq!(row(&ledger).1, blocks * P);
    // H/2 free: the asks take what streaming gives, down to the slots.
    let freed = std::thread::scope(|t| t.spawn(|| ledger.set_foreign(reading(&ledger, varena::headroom(CARD) / 2), false).unwrap()).join().unwrap());
    let left = row(&ledger).1;
    assert!(freed > 0 && left < blocks * P, "the running set gave nothing: freed {freed}, left {left}");
    for b in 0..blocks as u32 {
        let pin = mem.bind(s, b, 0).unwrap_or_else(|e| panic!("block {b} after the shrink: {e}"));
        // SAFETY: a host page mapped at `ptr`, pinned for this read.
        assert!(unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) }.iter().all(|x| *x == 5));
    }
    pass.end(0);
    assert_eq!(r.end().leftover, 0);
}

/// At the floor, the refusal names the foreign process TensorD saw.
#[test]
fn a_refusal_at_the_floor_names_the_foreign_process() {
    let (_scope, g) = Scope::new("name");
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    ledger.set_foreign_holder(Some(("game.x86_64", 3 << 30))).unwrap();
    ledger.set_foreign(CARD, false).unwrap();
    let r = mem.request("r", Plan::default()).unwrap();
    let e = mem.make_room(P).map(drop).expect_err("no room at all");
    let varena::Error::Exhausted(x) = &e else { panic!("{e}") };
    assert_eq!(x.foreign.as_deref(), Some("game.x86_64 (3072 MiB)"), "{e}");
    assert!(e.to_string().contains("GPU memory in use by game.x86_64"), "{e}");
    drop(r);
}

/// Growth only at a pass boundary and only beyond 2H free on the device: a streamed set inside the
/// H..2H band does not grow; with three blocks beyond 2H, the boundary takes back three.
#[test]
fn growth_waits_for_a_pass_boundary_and_2h_free() {
    let (_scope, g) = Scope::new("grow");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    let blocks = 10;
    let s = mem.load(set(dir.path(), "run", blocks, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("run")).unwrap();
    let sweep = |r: &varena::Request| {
        let pass = r.pass(s, "k").unwrap();
        for b in 0..blocks as u32 {
            drop(mem.bind(s, b, 0).unwrap());
        }
        pass.end(0);
    };
    sweep(&r);
    let h = varena::headroom(CARD);
    // A foreign process leaves H/2: it streams.
    ledger.set_foreign(reading(&ledger, h / 2), false).unwrap();
    let streamed = row(&ledger).1;
    assert!(streamed < blocks * P, "it did not stream");
    // Room comes back, but the device stays inside the H..2H band: no growth at the boundary.
    ledger.set_foreign(reading(&ledger, 2 * h - P), false).unwrap();
    let pass = r.pass(s, "k").unwrap();
    assert_eq!(row(&ledger).1, streamed, "it grew inside the H..2H band");
    pass.end(0);
    // Three blocks beyond 2H: the next boundary takes back three.
    ledger.set_foreign(reading(&ledger, 2 * h + 3 * P), false).unwrap();
    let pass = r.pass(s, "k").unwrap();
    assert_eq!(row(&ledger).1, streamed + 3 * P, "the boundary took back three blocks");
    pass.end(0);
    drop(r);
}

/// Y0 (the reviewer's probe): a real reading, the device left with H/2 free, yields: shared mode
/// charges H, so the rule fires on NVML's figure, not a made-up one.
#[test]
fn a_reading_with_less_than_h_free_yields() {
    let (_scope, g) = Scope::new("y0");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "idle", 4, &g.uuid), Level::Gpu).unwrap();
    let freed = ledger.set_foreign(reading(&ledger, varena::headroom(CARD) / 2), false).unwrap();
    assert!(freed > 0, "device free H/2 < H, and nothing was asked");
    drop(mem);
}

/// Y1 (the reviewer's probe): a W ask mid-pass streams the running set down to its floor; the
/// same pass then binds every block again without growing back (slots refill at a boundary).
#[test]
fn no_growth_inside_the_pass_after_a_shrink() {
    let (_scope, g) = Scope::new("y1");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap());
    let blocks = 8;
    let s = mem.load(set(dir.path(), "run", blocks, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("run")).unwrap();
    let pass = r.pass(s, "k").unwrap();
    for b in 0..blocks as u32 {
        drop(mem.bind(s, b, 0).unwrap());
    }
    std::thread::scope(|t| t.spawn(|| ledger.set_foreign(reading(&ledger, varena::headroom(CARD) / 2), false).unwrap()).join().unwrap());
    let after = row(&ledger).1;
    let mut most = after;
    for b in 0..blocks as u32 {
        drop(mem.bind(s, b, 0).unwrap());
        most = most.max(row(&ledger).1);
    }
    pass.end(0);
    drop(r);
    assert_eq!(most, after, "the streamed set grew inside the pass");
}

/// Y3 (the reviewer's probe): a new foreign process takes a running row's idle pages (a set its
/// plan does not use), never its running set.
#[test]
fn a_new_process_takes_a_running_rows_idle_pages() {
    let (_scope, g) = Scope::new("y3");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "other", 3, &g.uuid), Level::Gpu).unwrap();
    let s = mem.load(set(dir.path(), "run", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("run")).unwrap();
    let pass = r.pass(s, "k").unwrap();
    drop((mem.bind(s, 0, 0).unwrap(), mem.bind(s, 1, 0).unwrap()));
    let freed = ledger.set_foreign(reading(&ledger, 2 * varena::headroom(CARD)), true).unwrap();
    pass.end(0);
    drop(r);
    assert_eq!(freed, 3 * P, "a running row's idle pages stayed for a new foreign process");
}

/// A row that answered nothing is not asked again until its classes change: at the floor, each
/// reading would ask it for nothing.
#[test]
fn a_row_that_gave_nothing_is_not_asked_again_until_it_changes() {
    let (_scope, g) = Scope::new("quiet");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    let s = mem.load(set(dir.path(), "run", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("run")).unwrap();
    let pass = r.pass(s, "k").unwrap();
    let pins = (mem.bind(s, 0, 0).unwrap(), mem.bind(s, 1, 0).unwrap());
    // Everything pinned: the first reading's asks get nothing, the second makes none.
    let short = || ledger.set_foreign(reading(&ledger, 0), false).unwrap();
    assert_eq!((short(), short()), (0, 0));
    let asked = ledger.status().rows.iter().map(|r| r.asked).sum::<u64>();
    drop(pins);
    pass.end(0);
    drop(r);
    assert_eq!(asked, 0);
}

/// A fake per-process budget (Windows' DXCore): `set` changes it and wakes the waiter; each
/// read is counted, so a test knows the waiter looked.
#[derive(Default)]
struct FakeBudget {
    now: Mutex<Now>,
    changed: Condvar,
}

#[derive(Default)]
struct Now {
    budget: u64,
    usage: u64,
    changes: u64,
    reads: u64,
    closed: bool,
}

impl FakeBudget {
    fn set(&self, budget: u64, usage: u64) -> u64 {
        let mut n = self.now.lock();
        (n.budget, n.usage, n.changes) = (budget, usage, n.changes + 1);
        self.changed.notify_all();
        n.reads
    }

    /// Until the waiter read past `reads` (hang protection only).
    fn read_past(&self, reads: u64) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut n = self.now.lock();
        while n.reads <= reads && !self.changed.wait_until(&mut n, deadline).timed_out() {}
        assert!(n.reads > reads, "the budget's thread never read the change");
    }
}

impl Budget for FakeBudget {
    fn read(&self) -> varena::Result<(u64, u64)> {
        let mut n = self.now.lock();
        n.reads += 1;
        self.changed.notify_all();
        Ok((n.budget, n.usage))
    }
    fn wait(&self) -> bool {
        let mut n = self.now.lock();
        let seen = n.changes;
        while n.changes == seen && !n.closed {
            self.changed.wait(&mut n);
        }
        !n.closed
    }
    fn close(&self) {
        self.now.lock().closed = true;
        self.changed.notify_all();
    }
}

/// Host pages under a fake budget.
struct Budgeted(HostBackend, Arc<FakeBudget>);

impl Backend for Budgeted {
    fn capacity(&self) -> u64 { self.0.capacity() }
    fn free(&self) -> varena::Result<u64> { self.0.free() }
    fn process_used(&self) -> varena::Result<u64> { self.0.process_used() }
    fn opened_context(&self) -> u64 { self.0.opened_context() }
    fn granularity(&self) -> u64 { self.0.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.0.reserve(len) }
    fn release(&self, va: Va) { self.0.release(va) }
    fn create(&self, len: u64) -> varena::Result<Physical> { self.0.create(len) }
    fn destroy(&self, page: Physical) -> varena::Result<()> { self.0.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> { self.0.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> { self.0.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> { self.0.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> varena::Result<Owned> { self.0.export(page) }
    fn import(&self, handle: Owned, len: u64) -> varena::Result<Physical> { self.0.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.0.target(va, offset, len) }
    fn fence(&self) -> varena::Result<Event> { self.0.fence() }
    fn record(&self, stream: usize) -> varena::Result<Event> { self.0.record(stream) }
    fn query(&self, event: Event) -> varena::Result<bool> { self.0.query(event) }
    fn elapsed(&self, from: Event, to: Event) -> varena::Result<Option<u64>> { self.0.elapsed(from, to) }
    fn wait(&self, event: Event, stream: usize) -> varena::Result<()> { self.0.wait(event, stream) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> { self.0.synchronize(stream) }
    fn finish(&self, event: Event) -> varena::Result<()> { self.0.finish(event) }
    fn retire(&self, event: Event) { self.0.retire(event) }
    fn drain(&self) -> varena::Result<()> { self.0.drain() }
    fn budget(&self) -> Option<Arc<dyn Budget>> { Some(self.1.clone()) }
}

/// Windows: a budget change that leaves usage above budget − H gives back what brings it to
/// budget − 2H, from the budget's own thread; one within the band gives nothing. A refusal names
/// the budget.
#[test]
fn a_budget_change_gives_back_down_to_budget_less_2h() {
    let (_scope, g) = Scope::new("budget");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let budget = Arc::new(FakeBudget::default());
    let h = varena::headroom(CARD);
    budget.set(CARD, 0);
    let mem = Memory::open(g.clone(), None, Box::new(Budgeted(HostBackend::new(CARD), budget.clone()))).unwrap();
    mem.load(set(dir.path(), "idle", 4, &g.uuid), Level::Gpu).unwrap();
    let used = 4 * P;
    // Inside the band (usage ≤ budget − H): read, nothing given.
    let reads = budget.set(used + h, used);
    budget.read_past(reads);
    mem.settle();
    assert_eq!(row(&ledger).1, used, "a change inside the band gave pages back");
    // Over it by two pages: back to budget − 2H takes those and H more, so every idle page.
    let reads = budget.set(used + h - 2 * P, used);
    budget.read_past(reads);
    mem.settle();
    assert_eq!(row(&ledger).1, 0, "the budget's thread gave the idle pages back");
    // At the floor (nothing left to give), a refusal names the budget.
    ledger.set_foreign(CARD, false).unwrap();
    let r = mem.request("r", Plan::default()).unwrap();
    let e = mem.make_room(P).map(drop).expect_err("refused");
    assert!(e.to_string().contains("budget is"), "{e}");
    drop(r);
}

/// A standalone executor's fixed row (varena-leases R2-2): its bytes are held, so the card's free
/// room counts them, and no reading asks them, however short the card is; dropped, the row goes.
#[test]
fn a_fixed_row_is_held_and_never_asked() {
    let (_scope, g) = Scope::new("fixed");
    let dir = tempfile::tempdir().unwrap();
    let ledger = Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(CARD))).unwrap();
    mem.load(set(dir.path(), "idle", 2, &g.uuid), Level::Gpu).unwrap();
    let free = ledger.status().free;
    let (pid, standalone) = (std::process::id() + 1, 4 << 30);
    let fixed = ledger.fixed(pid).unwrap();
    fixed.set(standalone);
    let held = |ledger: &Ledger| ledger.status().rows.iter().find(|r| r.pid == pid && r.live).map(|r| r.context);
    assert_eq!((held(&ledger), ledger.status().free), (Some(standalone), free - standalone));
    let freed = ledger.set_foreign(CARD, false).unwrap();
    assert_eq!((freed, row(&ledger).1, held(&ledger)), (2 * P, 0, Some(standalone)), "only the idle pages are asked");
    ledger.set_foreign(0, false).unwrap();
    drop(fixed);
    assert_eq!((held(&ledger), ledger.status().free), (None, free + 2 * P));
    drop(mem);
}
