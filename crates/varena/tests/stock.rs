//! The allocator callback's stock (S1 review 5, F1-F6): the callback claims only pages the state
//! offered, never takes the state lock, never takes a page a plan, pass or bind made W, D or P,
//! and its reclaims and pins' drops wake waiters. The reviewer's probes, as tests. Run under
//! `unshare --user --net` like the rest (each test names its own namespace).
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Measured, Memory, Origin, Physical, Plan, Result, SetSpec, Step, Torch, Va};

fn spec(dir: &std::path::Path, uuid: &str, name: &str, fill: Fill) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![7u8; REGION_ALIGN as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: REGION_ALIGN };
    let layout = Layout {
        parts: vec![Part { what: "t0".into(), region: 0, offset: 0, nbytes: REGION_ALIGN }],
        regions: vec![Region { offset: 0, nbytes: REGION_ALIGN, span: REGION_ALIGN, items: vec![Item { offset: 0, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: 0, len: REGION_ALIGN }), part_end: true }] }],
        nbytes: REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: vec![vec!["t0".into()]],
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], fill, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

fn plan(set: &str) -> Plan {
    Plan { steps: vec![Step { sets: vec![set.into()], passes: 1 }] }
}

/// Use a set's one block in a request of its own and let it go: the page is filled, idle and
/// of no declared use (N), so the next publication offers it to the allocator callback.
fn use_once(mem: &Memory, set: varena::SetId, name: &str, req: &str) {
    let r = mem.request(req, plan(name)).unwrap();
    let pass = r.pass(set, "fwd", "1x1", Torch::default()).unwrap();
    let pin = mem.bind(set, 0, 0).unwrap();
    if pin.fresh {
        pin.filled();
    }
    drop(pin);
    pass.end(Measured::default(), Torch::default());
    r.end();
    mem.torch_limit();
}

#[derive(Default)]
struct Gate {
    /// (open, a call is held at it)
    state: Mutex<(bool, bool)>,
    cv: Condvar,
}

impl Gate {
    fn set(&self, v: bool) {
        self.state.lock().0 = v;
        self.cv.notify_all();
    }
    /// A call waits here until the gate opens; `entered` sees it arrive.
    fn pass(&self) {
        let mut s = self.state.lock();
        s.1 = true;
        self.cv.notify_all();
        while !s.0 {
            self.cv.wait(&mut s);
        }
    }
    fn entered(&self) {
        let mut s = self.state.lock();
        while !s.1 {
            self.cv.wait(&mut s);
        }
    }
}

/// Host pages whose unmaps can be held (a card's cuMemUnmap waits for queued device work).
struct Held {
    host: HostBackend,
    hold: Arc<AtomicBool>,
    held: Arc<AtomicU32>,
    gate: Arc<Gate>,
}

impl Backend for Held {
    fn capacity(&self) -> u64 { self.host.capacity() }
    fn free(&self) -> Result<u64> { self.host.free() }
    fn process_used(&self) -> Result<u64> { self.host.process_used() }
    fn opened_context(&self) -> u64 { self.host.opened_context() }
    fn granularity(&self) -> u64 { self.host.granularity() }
    fn reserve(&self, len: u64) -> Result<Va> { self.host.reserve(len) }
    fn release(&self, va: Va) { self.host.release(va) }
    fn create(&self, len: u64) -> Result<Physical> { self.host.create(len) }
    fn destroy(&self, page: Physical) -> Result<()> { self.host.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> { self.host.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> { self.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        if self.hold.load(Ordering::SeqCst) {
            self.held.fetch_add(1, Ordering::SeqCst);
            self.gate.pass();
        }
        self.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> { self.host.export(page) }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> { self.host.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.host.target(va, offset, len) }
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> Result<()> { self.host.save(va, offset, host, after) }
    fn fence(&self) -> Result<Event> { self.host.fence() }
    fn record(&self, stream: usize) -> Result<Event> { self.host.record(stream) }
    fn query(&self, event: Event) -> Result<bool> { self.host.query(event) }
    fn elapsed(&self, a: Event, b: Event) -> Result<Option<u64>> { self.host.elapsed(a, b) }
    fn wait(&self, event: Event, stream: usize) -> Result<()> { self.host.wait(event, stream) }
    fn synchronize(&self, stream: usize) -> Result<()> { self.host.synchronize(stream) }
    fn finish(&self, event: Event) -> Result<()> { self.host.finish(event) }
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> { self.host.register_host(ptr, len) }
    fn unregister_host(&self, ptr: u64) -> Result<()> { self.host.unregister_host(ptr) }
    fn retire(&self, event: Event) { self.host.retire(event) }
    fn drain(&self) -> Result<()> { self.host.drain() }
    fn drain_save(&self) -> Result<()> { self.host.drain_save() }
}

type Hook = Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>;

/// Host pages with release events that stay pending until armed; the first armed query runs
/// `hook` on the querying thread (of a pending event, or of any event when `every`).
struct Hooked {
    host: HostBackend,
    next: std::sync::atomic::AtomicU64,
    pending: Arc<AtomicBool>,
    armed: Arc<AtomicBool>,
    hook: Hook,
    every: bool,
}

impl Backend for Hooked {
    fn capacity(&self) -> u64 { self.host.capacity() }
    fn free(&self) -> Result<u64> { self.host.free() }
    fn process_used(&self) -> Result<u64> { self.host.process_used() }
    fn opened_context(&self) -> u64 { self.host.opened_context() }
    fn granularity(&self) -> u64 { self.host.granularity() }
    fn reserve(&self, len: u64) -> Result<Va> { self.host.reserve(len) }
    fn release(&self, va: Va) { self.host.release(va) }
    fn create(&self, len: u64) -> Result<Physical> { self.host.create(len) }
    fn destroy(&self, page: Physical) -> Result<()> { self.host.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> { self.host.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> { self.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> { self.host.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> Result<Owned> { self.host.export(page) }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> { self.host.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.host.target(va, offset, len) }
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> Result<()> { self.host.save(va, offset, host, after) }
    fn fence(&self) -> Result<Event> { self.host.fence() }
    fn record(&self, _: usize) -> Result<Event> {
        Ok(if self.pending.load(Ordering::SeqCst) { Event(1000 + self.next.fetch_add(1, Ordering::SeqCst)) } else { Event::DONE })
    }
    fn query(&self, event: Event) -> Result<bool> {
        if !self.armed.load(Ordering::SeqCst) {
            return Ok(event == Event::DONE);
        }
        if event == Event::DONE && !self.every {
            return Ok(true);
        }
        if let Some(hook) = self.hook.lock().take() {
            hook();
        }
        Ok(true)
    }
    fn elapsed(&self, a: Event, b: Event) -> Result<Option<u64>> { self.host.elapsed(a, b) }
    fn wait(&self, _: Event, _: usize) -> Result<()> { Ok(()) }
    fn synchronize(&self, stream: usize) -> Result<()> { self.host.synchronize(stream) }
    fn finish(&self, _: Event) -> Result<()> { Ok(()) }
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> { self.host.register_host(ptr, len) }
    fn unregister_host(&self, ptr: u64) -> Result<()> { self.host.unregister_host(ptr) }
    fn retire(&self, _: Event) {}
    fn drain(&self) -> Result<()> { self.host.drain() }
    fn drain_save(&self) -> Result<()> { self.host.drain_save() }
}


/// F2: an offer made while a page had no declared use (N) is taken back when a request makes it
/// its working set (W): the callback never reclaims the running pass's own page. Both sets are
/// measured first (an unmeasured pass sweeps idle pages itself, ruling 1), and the other set's
/// page is pinned, so the stale offer is the only page the stock could hold.
#[test]
fn a_stale_offer_never_gives_the_callback_a_working_set_page() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stockstale-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let w = mem.load(spec(dir.path(), &uuid, "wset", Fill::Undeclared), Level::Ram).unwrap();
    let x = mem.load(spec(dir.path(), &uuid, "xset", Fill::Undeclared), Level::Ram).unwrap();
    for n in 0..2 {
        use_once(&mem, w, "wset", &format!("w{n}"));
        use_once(&mem, x, "xset", &format!("x{n}"));
    }
    let px = mem.bind(x, 0, 0).unwrap();
    let r2 = mem.request("r2", plan("wset")).unwrap();
    let pass = r2.pass(w, "fwd", "1x1", Torch::default()).unwrap();
    let freed = r2.reclaim_allocation(REGION_ALIGN).unwrap();
    drop(mem.bind(w, 0, 0).unwrap());
    let origin = r2.stats().trace.last().map(|t| t.origin);
    drop(pass);
    r2.end();
    drop(px);
    drop(mem);
    varena::os::remove_table(&uuid);
    assert_eq!(freed, 0, "the callback reclaimed a page");
    assert_eq!(origin, Some(Origin::Hit), "the callback reclaimed the running pass's working-set page");
}

/// F1, F4: the callback never takes the state lock: while its unmap waits (as cuMemUnmap may,
/// behind queued device work), every other state call goes on.
#[test]
fn the_callback_holds_no_state_lock_while_it_unmaps() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stocklock-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let (hold, held, gate) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU32::new(0)), Arc::new(Gate::default()));
    let backend = Held { host: HostBackend::new(g.memory), hold: hold.clone(), held: held.clone(), gate: gate.clone() };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let a = mem.load(spec(dir.path(), &uuid, "aset", Fill::Undeclared), Level::Ram).unwrap();
    let n = mem.load(spec(dir.path(), &uuid, "nset", Fill::Undeclared), Level::Ram).unwrap();
    use_once(&mem, n, "nset", "r0");
    let r = Arc::new(mem.request("r", plan("aset")).unwrap());
    let pin = mem.bind(a, 0, 0).unwrap();
    hold.store(true, Ordering::SeqCst);
    let reclaim = std::thread::spawn({
        let r = r.clone();
        move || r.reclaim_allocation(REGION_ALIGN)
    });
    gate.entered();
    // Hang protection only: a state call while the callback's unmap is held.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn({
        let mem = mem.clone();
        move || tx.send(mem.torch_limit()).unwrap()
    });
    let other = rx.recv_timeout(Duration::from_secs(10));
    gate.set(true);
    hold.store(false, Ordering::SeqCst);
    let freed = reclaim.join().unwrap().unwrap();
    drop(pin);
    drop(r);
    drop(mem);
    varena::os::remove_table(&uuid);
    assert!(other.is_ok(), "a state call waited for the callback's unmap");
    assert_eq!(freed, REGION_ALIGN, "the callback reclaimed the offered page");
}

/// F5: a reconcile moves the callback's reclaim into the stored limit: the allocator's limit
/// never drops below the state's between publications, so a retry sees the room.
#[test]
fn a_reconcile_keeps_the_allocator_limit() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stocklimit-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    mem.load(spec(dir.path(), &uuid, "aset", Fill::Undeclared), Level::Ram).unwrap();
    let x = mem.load(spec(dir.path(), &uuid, "xset", Fill::Undeclared), Level::Ram).unwrap();
    use_once(&mem, x, "xset", "r0");
    let r = mem.request("r", plan("aset")).unwrap();
    let base = mem.allocator_limit().unwrap();
    let freed = r.reclaim_allocation(REGION_ALIGN).unwrap();
    let claimed = mem.allocator_limit().unwrap();
    let truth = mem.torch_limit();
    let after = mem.allocator_limit().unwrap();
    r.end();
    drop(mem);
    varena::os::remove_table(&uuid);
    assert_eq!((freed, claimed), (REGION_ALIGN, base + REGION_ALIGN), "the reclaim counts at once");
    assert!(after >= truth, "the allocator's limit {after} fell below the state's {truth} after a reconcile");
}

/// Constraint 1 (review 5): with the stock the callback's only source, a plan that makes a page
/// N offers it again (and a plan that makes it D takes it back).
#[test]
fn a_new_plan_offers_a_page_it_made_idle() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stockplan-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    let p = mem.load(spec(dir.path(), &uuid, "pset", Fill::Undeclared), Level::Ram).unwrap();
    mem.load(spec(dir.path(), &uuid, "oset", Fill::Undeclared), Level::Ram).unwrap();
    use_once(&mem, p, "pset", "r0");
    // Declared by a request's plan: D, not the callback's.
    let r = mem.request("r", plan("pset")).unwrap();
    let declared = r.reclaim_allocation(REGION_ALIGN).unwrap();
    // A new plan without it: N again, offered.
    r.plan(plan("oset"));
    let idle = r.reclaim_allocation(REGION_ALIGN).unwrap();
    r.end();
    drop(mem);
    varena::os::remove_table(&uuid);
    assert_eq!(declared, 0, "the callback took a page a plan declared");
    assert_eq!(idle, REGION_ALIGN, "a page a new plan made idle was not offered");
}

/// F3: the callback's reclaim (no state lock) lands after the binder's last check and before it
/// sleeps (run inside the binder's own drain here): the binder still wakes for it.
#[test]
fn a_reclaim_that_lands_before_the_binder_waits_wakes_it() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stockwake-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let (pending, armed) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let hook: Hook = Arc::new(Mutex::new(None));
    let backend = Hooked { host: HostBackend::new(g.memory), next: Default::default(), pending: pending.clone(), armed: armed.clone(), hook: hook.clone(), every: false };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let k = mem.load(spec(dir.path(), &uuid, "kset", Fill::Undeclared), Level::Ram).unwrap();
    let j = mem.load(spec(dir.path(), &uuid, "jset", Fill::Undeclared), Level::Ram).unwrap();
    let l = mem.load(spec(dir.path(), &uuid, "lset", Fill::Undeclared), Level::Ram).unwrap();
    use_once(&mem, k, "kset", "r0");
    let r = Arc::new(mem.request("r", plan("lset")).unwrap());
    let wake = mem.bind(l, 0, 0).unwrap();
    if wake.fresh {
        wake.filled();
    }
    let pj = mem.bind(j, 0, 0).unwrap();
    if pj.fresh {
        pj.filled();
    }
    pending.store(true, Ordering::SeqCst);
    drop(pj);
    pending.store(false, Ordering::SeqCst);
    let claimed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    *hook.lock() = Some(Box::new({
        let (r, claimed) = (r.clone(), claimed.clone());
        move || claimed.store(r.reclaim_allocation(REGION_ALIGN).unwrap_or(0), Ordering::SeqCst)
    }));
    armed.store(true, Ordering::SeqCst);
    let (tx, rx) = std::sync::mpsc::channel();
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || {
            let pin = mem.bind(k, 0, 0);
            if let Ok(p) = &pin {
                if p.fresh {
                    p.filled();
                }
            }
            tx.send(pin.is_ok()).unwrap();
        }
    });
    // Hang protection only: a lost wake sleeps until the unrelated pin below is dropped.
    let first = rx.recv_timeout(Duration::from_secs(10));
    drop(wake);
    let second = if first.is_err() { rx.recv_timeout(Duration::from_secs(10)) } else { first };
    binder.join().unwrap();
    drop(r);
    drop(mem);
    varena::os::remove_table(&uuid);
    assert_eq!(claimed.load(Ordering::SeqCst), REGION_ALIGN, "the callback claimed the page");
    assert!(first.is_ok(), "the binder slept through the callback's wake ({second:?} after an unrelated one)");
}

/// F3: a pin's drop (no state lock: a DLPack deleter may hold the GIL or the allocator lock)
/// lands after the binder's last check and before it sleeps: the binder still wakes for it.
#[test]
fn a_pin_dropped_before_the_binder_waits_wakes_it() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stockpin-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let (pending, armed) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let hook: Hook = Arc::new(Mutex::new(None));
    let backend = Hooked { host: HostBackend::new(g.memory), next: Default::default(), pending: pending.clone(), armed: armed.clone(), hook: hook.clone(), every: false };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let k = mem.load(spec(dir.path(), &uuid, "kset", Fill::Undeclared), Level::Ram).unwrap();
    let j = mem.load(spec(dir.path(), &uuid, "jset", Fill::Undeclared), Level::Ram).unwrap();
    let l = mem.load(spec(dir.path(), &uuid, "lset", Fill::Undeclared), Level::Ram).unwrap();
    let wake = mem.bind(l, 0, 0).unwrap();
    if wake.fresh {
        wake.filled();
    }
    let a_pin = mem.bind(k, 0, 0).unwrap();
    if a_pin.fresh {
        a_pin.filled();
    }
    let pj = mem.bind(j, 0, 0).unwrap();
    if pj.fresh {
        pj.filled();
    }
    pending.store(true, Ordering::SeqCst);
    drop(pj);
    pending.store(false, Ordering::SeqCst);
    mem.fill(k, Fill::Declared("bake-2".into()));
    *hook.lock() = Some(Box::new(move || drop(a_pin)));
    armed.store(true, Ordering::SeqCst);
    let (tx, rx) = std::sync::mpsc::channel();
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || {
            let pin = mem.bind(k, 0, 0);
            if let Ok(p) = &pin {
                if p.fresh {
                    p.filled();
                }
            }
            tx.send(pin.map(|_| ()).map_err(|e| e.to_string())).unwrap();
        }
    });
    // Hang protection only: a lost wake sleeps until the unrelated pin below is dropped.
    let first = rx.recv_timeout(Duration::from_secs(10));
    drop(wake);
    let second = if first.is_err() { rx.recv_timeout(Duration::from_secs(10)) } else { first.clone() };
    binder.join().unwrap();
    drop(mem);
    varena::os::remove_table(&uuid);
    assert!(first.is_ok(), "the binder slept through the pin drop's wake ({second:?} after an unrelated one)");
}

/// Review 6, G1 (the reviewer's probe): a restock re-checks the live offers in place, so an
/// allocator callback during a restock finds the idle N page offered (it used to find the
/// stock empty: the offers were taken out and sent back after). A live offer's re-check makes
/// no driver call (G2), so the callback runs inside the restock's one call, a fresh offer's
/// event query, standing in for the allocator on another thread.
#[test]
fn a_restock_never_leaves_the_callback_an_empty_stock() {
    assert!(varena::os::namespace(&format!("stock{}-", std::process::id())));
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("rv8restock-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let (pending, armed) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let hook: Hook = Arc::new(Mutex::new(None));
    let backend = Hooked { host: HostBackend::new(g.memory), next: Default::default(), pending, armed: armed.clone(), hook: hook.clone(), every: true };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let k = mem.load(spec(dir.path(), &uuid, "kset", Fill::Undeclared), Level::Ram).unwrap();
    mem.load(spec(dir.path(), &uuid, "other", Fill::Undeclared), Level::Ram).unwrap();
    let third = mem.load(spec(dir.path(), &uuid, "third", Fill::Undeclared), Level::Ram).unwrap();
    // K used and let go: offered to the callback as N.
    use_once(&mem, k, "kset", "r0");
    let r1 = Arc::new(mem.request("r1", plan("other")).unwrap());
    // The third set's page in use, then N once its request ends: a fresh offer then.
    let r3 = mem.request("r3", plan("third")).unwrap();
    let pass = r3.pass(third, "fwd", "1x1", Torch::default()).unwrap();
    let pin = mem.bind(third, 0, 0).unwrap();
    if pin.fresh {
        pin.filled();
    }
    drop(pin);
    pass.end(Measured::default(), Torch::default());
    let claimed = Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
    *hook.lock() = Some(Box::new({
        let (r1, claimed) = (r1.clone(), claimed.clone());
        move || {
            let _ = claimed.compare_exchange(u64::MAX, r1.reclaim_allocation(REGION_ALIGN).unwrap_or(0), Ordering::SeqCst, Ordering::SeqCst);
        }
    }));
    armed.store(true, Ordering::SeqCst);
    // Its end restocks: K's live offer is re-checked, the third set's page offered fresh.
    r3.end();
    armed.store(false, Ordering::SeqCst);
    let during = claimed.load(Ordering::SeqCst);
    println!("callback during the restock claimed {during}");
    drop(r1);
    drop(mem);
    varena::os::remove_table(&uuid);
    assert_ne!(during, u64::MAX, "the hook never ran: the probe tests nothing");
    assert_eq!(during, REGION_ALIGN, "the callback found the stock empty while an idle N page was offered");
}
