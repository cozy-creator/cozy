//! Ordering against device events: a pin's release is not complete before its event, a page
//! goes only after its last consumer, and no host thread waits on a device event. Host pages
//! fill synchronously, so the
//! backend here is the host backend with stream events that complete only when the test says
//! (as a card's do when their work finishes), and with the stream waits it was asked for.
// Its sets load from TensorFS sources, which are Unix's so far (`tensorfs::io::source`).
#![cfg(unix)]

use std::collections::HashSet;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Plan, Result, SetSpec, Step, Va};

#[derive(Default)]
struct Events {
    next: AtomicU64,
    done: Mutex<HashSet<u64>>,
    /// Event queries made.
    queries: AtomicU64,
    /// Calls that make a page read-only or hand it out, and host waits on an event ("finish"),
    /// each signalled.
    kept: Mutex<Vec<&'static str>>,
    logged: Condvar,
    unmaps: AtomicU64,
    /// In-stream waits asked for, in order: (event, stream).
    waits: Mutex<Vec<(u64, usize)>>,
    /// Fences are pending events too (a card's copy), and a fence may hold its fill thread
    /// until a gate opens (a long copy).
    pending_fences: AtomicBool,
    fences: Mutex<Vec<u64>>,
    hold_fence: AtomicBool,
    fence_gate: Gate,
    /// The next event record fails (a driver error); a deleter must never synchronize.
    fail_record: AtomicBool,
    fail_finish: AtomicBool,
    fail_unmap: AtomicBool,
    fail_destroy: AtomicBool,
    syncs: Mutex<Vec<usize>>,
    /// Another process's bytes on the device, and bytes that land there at the next create
    /// (between a decision and its page).
    foreign: AtomicU64,
    lands: AtomicU64,
    /// Driver allocation can fail despite its reported free bytes (for example metadata room).
    hidden_ooms: AtomicU64,
    creates: AtomicU64,
    /// The process's own bytes cannot be measured (WDDM); open measured this context.
    unmeasured: AtomicU64,
    /// The next unmap waits (as a card's does on queued work) until the gate opens.
    hold_unmap: AtomicBool,
    unmap_gate: Gate,
    /// Opened when a held unmap is waiting.
    unmap_waits: Gate,
    /// Signalled when the test completes an event.
    completed: Condvar,
    /// Events varena gave up (retired), and a signal for each.
    retired: Mutex<HashSet<u64>>,
    retiring: Condvar,
    /// Mapping calls by address, in order: map, target (a fill writes), unmap.
    calls: Mutex<Vec<(&'static str, u64)>>,
    /// The next map or fill target waits until its gate opens (its `waits` opens then).
    hold_map: AtomicBool,
    map_gate: Gate,
    map_waits: Gate,
    hold_target: AtomicBool,
    target_gate: Gate,
    target_waits: Gate,
    /// The next driver free-bytes read waits (under the state lock, as refresh reads it).
    hold_free: AtomicBool,
    free_gate: Gate,
    free_waits: Gate,
    /// The next protect waits until its gate opens (its `waits` opens then); a protect is
    /// running; binds (a binder's in-stream wait) made while one ran; each protect's bytes.
    hold_protect: AtomicBool,
    protect_gate: Gate,
    protect_waits: Gate,
    protecting: AtomicBool,
    bound_while_protecting: AtomicU64,
    protected: Mutex<Vec<u64>>,
}

/// Holds the call when `hold` is set: `waits` opens, then it passes `gate`.
fn held(hold: &AtomicBool, waits: &Gate, gate: &Gate) {
    if hold.swap(false, Ordering::SeqCst) {
        waits.open();
        gate.pass();
    }
}

impl Events {
    /// The work behind `event` finished, as a card's does.
    fn complete(&self, event: u64) {
        self.done.lock().insert(event);
        self.completed.notify_all();
    }

    fn log(&self, call: &'static str) {
        self.kept.lock().push(call);
        self.logged.notify_all();
    }

    /// An event used after its retire is an error, as the CUDA backend's (its id is gone).
    fn live(&self, event: Event, op: &'static str) -> Result<()> {
        match event != Event::DONE && self.retired.lock().contains(&event.0) {
            true => Err(varena::Error::Device { op, code: 400, name: format!("event {} used after its retire", event.0) }),
            false => Ok(()),
        }
    }

    /// Blocks until the log of calls satisfies `until`.
    fn await_log(&self, until: impl Fn(&[&'static str]) -> bool) {
        let mut log = self.kept.lock();
        while !until(&log) {
            self.logged.wait(&mut log);
        }
    }

}

/// Closed until opened.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    cv: Condvar,
}

impl Gate {
    fn open(&self) {
        *self.open.lock() = true;
        self.cv.notify_all();
    }
    fn pass(&self) {
        let mut open = self.open.lock();
        while !*open {
            self.cv.wait(&mut open);
        }
    }
}

struct Stream {
    host: HostBackend,
    events: Arc<Events>,
}

impl Backend for Stream {
    fn capacity(&self) -> u64 {
        self.host.capacity()
    }
    fn headroom(&self) -> u64 {
        self.host.headroom()
    }
    fn free(&self) -> Result<u64> {
        held(&self.events.hold_free, &self.events.free_waits, &self.events.free_gate);
        Ok(self.host.free()?.saturating_sub(self.events.foreign.load(Ordering::SeqCst)))
    }
    fn process_used(&self) -> Result<u64> {
        match self.events.unmeasured.load(Ordering::SeqCst) {
            0 => self.host.process_used(),
            _ => Err(varena::Error::Device { op: "process_used", code: 999, name: "not measurable".into() }),
        }
    }
    fn opened_context(&self) -> u64 {
        self.events.unmeasured.load(Ordering::SeqCst)
    }
    fn granularity(&self) -> u64 {
        self.host.granularity()
    }
    fn reserve(&self, len: u64) -> Result<Va> {
        self.host.reserve(len)
    }
    fn release(&self, va: Va) {
        self.host.release(va)
    }
    fn create(&self, len: u64) -> Result<Physical> {
        self.events.creates.fetch_add(1, Ordering::SeqCst);
        if self.events.hidden_ooms.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1)).is_ok() {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "allocation needs room beyond reported free bytes".into() });
        }
        let landed = self.events.lands.swap(0, Ordering::SeqCst);
        let foreign = self.events.foreign.fetch_add(landed, Ordering::SeqCst) + landed;
        if self.host.free()? < len + foreign {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        self.host.create(len)
    }
    fn destroy(&self, page: Physical) -> Result<()> {
        if self.events.fail_destroy.swap(false, Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "destroy", code: 999, name: "physical release unknown".into() });
        }
        self.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        held(&self.events.hold_map, &self.events.map_waits, &self.events.map_gate);
        self.host.map(va, offset, page, access)?;
        self.events.calls.lock().push(("map", va.base + offset));
        Ok(())
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        self.events.log("protect");
        self.events.protected.lock().push(len);
        self.events.protecting.store(true, Ordering::SeqCst);
        held(&self.events.hold_protect, &self.events.protect_waits, &self.events.protect_gate);
        let out = self.host.protect(va, offset, len, access);
        self.events.protecting.store(false, Ordering::SeqCst);
        out
    }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        if self.events.fail_unmap.swap(false, Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "unmap", code: 999, name: "mapping release unknown".into() });
        }
        if self.events.hold_unmap.swap(false, Ordering::SeqCst) {
            self.events.unmap_waits.open();
            self.events.unmap_gate.pass();
        }
        self.events.unmaps.fetch_add(1, Ordering::Relaxed);
        self.events.calls.lock().push(("unmap", va.base + offset));
        self.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> {
        self.events.log("export");
        self.host.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        self.host.import(handle, len)
    }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
        self.events.calls.lock().push(("target", va.base + offset));
        held(&self.events.hold_target, &self.events.target_waits, &self.events.target_gate);
        self.host.target(va, offset, len)
    }
    fn fence(&self) -> Result<Event> {
        if self.events.hold_fence.swap(false, Ordering::SeqCst) {
            self.events.fence_gate.pass();
        }
        if !self.events.pending_fences.load(Ordering::SeqCst) {
            return Ok(Event::DONE);
        }
        let e = self.events.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.events.fences.lock().push(e);
        Ok(Event(e))
    }
    /// A stream event: pending until the test completes it.
    fn record(&self, _: usize) -> Result<Event> {
        if self.events.fail_record.swap(false, Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "record", code: 999, name: "CUDA_ERROR_UNKNOWN".into() });
        }
        Ok(Event(self.events.next.fetch_add(1, Ordering::Relaxed) + 1))
    }
    fn query(&self, event: Event) -> Result<bool> {
        self.events.queries.fetch_add(1, Ordering::SeqCst);
        self.events.live(event, "query")?;
        Ok(event == Event::DONE || self.events.done.lock().contains(&event.0))
    }
    fn elapsed(&self, _: Event, _: Event) -> Result<Option<u64>> {
        Ok(Some(0))
    }
    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        // A binder's stream (a fill's copy stream is 0): a pin handed out under a protect.
        if stream != 0 && self.events.protecting.load(Ordering::SeqCst) {
            self.events.bound_while_protecting.fetch_add(1, Ordering::SeqCst);
        }
        self.events.live(event, "wait")?;
        if event != Event::DONE {
            self.events.waits.lock().push((event.0, stream));
        }
        Ok(())
    }
    fn synchronize(&self, stream: usize) -> Result<()> {
        self.events.syncs.lock().push(stream);
        Ok(())
    }
    /// Waits, as a card's host does, until the test completes the event.
    fn finish(&self, event: Event) -> Result<()> {
        self.events.log("finish");
        if self.events.fail_finish.swap(false, Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "finish", code: 999, name: "consumer completion unknown".into() });
        }
        self.events.live(event, "finish")?;
        let mut done = self.events.done.lock();
        while event != Event::DONE && !done.contains(&event.0) {
            self.events.completed.wait(&mut done);
        }
        Ok(())
    }
    fn retire(&self, event: Event) {
        let first = self.events.retired.lock().insert(event.0);
        assert!(first || event == Event::DONE, "event {} retired twice", event.0);
        self.events.retiring.notify_all();
    }
    fn drain(&self) -> Result<()> {
        Ok(())
    }
}

fn set(dir: &std::path::Path, uuid: &str) -> SetSpec {
    let path = dir.join("lora.bin");
    std::fs::write(&path, vec![7u8; REGION_ALIGN as usize]).unwrap();
    let obj = ObjectRef { sha256: "obj-lora".into(), length: REGION_ALIGN };
    let layout = Layout {
        parts: vec![Part { what: "t0".into(), region: 0, offset: 0, nbytes: REGION_ALIGN }],
        regions: vec![Region {
            offset: 0,
            nbytes: REGION_ALIGN,
            span: REGION_ALIGN,
            items: vec![Item { offset: 0, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: 0, len: REGION_ALIGN }), part_end: true }],
        }],
        nbytes: REGION_ALIGN,
        digest: format!("layout-lora-{uuid}"),
        names: vec![vec!["t0".into()]],
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Declared("lora-a".into()), "release-1", "0/1", uuid);
    SetSpec { name: "lora".into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Review 1, finding 12: a page released with its transform undone goes, but only after the
/// release event of the stream that last read it completes.
#[test]
fn a_page_unmaps_only_after_its_last_consumer() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("consumer-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Memory::open(g.clone(), Box::new(backend)).unwrap();
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["lora".into()], passes: 1 }] }).unwrap();
    let _pass = r.pass(id, "k").unwrap();
    let pin = mem.bind(id, 0, 1).unwrap();
    assert!(pin.fresh);
    drop(pin);
    let release = events.next.load(Ordering::Relaxed);
    for _ in 0..3 {
        mem.classes();
    }
    mem.settle();
    assert_eq!(events.unmaps.load(Ordering::Relaxed), 0, "unmapped while its consumer's stream ran");
    events.complete(release);
    mem.classes();
    mem.settle();
    assert_eq!(events.unmaps.load(Ordering::Relaxed), 1, "gone once the consumer finished");
    let again = mem.bind(id, 0, 1).unwrap();
    assert!(again.fresh, "filled again, fresh for the transform");
}

/// Round 2, the GPU bind hang (run 12818): a binder of a page whose release event is pending
/// never waits on the host for it. The page is filled again in place: the copy stream waits on
/// the release in-stream, and the binder's stream on the fill. The event completes later on
/// another thread, and no other varena call happens meanwhile.
#[test]
fn a_binder_never_waits_on_a_release_event() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("hang-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Arc::new(Memory::open(g.clone(), Box::new(backend)).unwrap());
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["lora".into()], passes: 1 }] }).unwrap();
    let _pass = r.pass(id, "k").unwrap();
    // Released fresh with its transform undone: its bytes are not the weights, and the stream
    // that read it has not finished.
    drop(mem.bind(id, 0, 1).unwrap());
    let release = events.next.load(Ordering::Relaxed);
    events.pending_fences.store(true, Ordering::SeqCst);
    let gate = Arc::new(Gate::default());
    let completes = {
        let (gate, events) = (gate.clone(), events.clone());
        std::thread::spawn(move || {
            gate.pass();
            events.complete(release);
        })
    };
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || {
            let pin = mem.bind(id, 0, 2).unwrap();
            (pin.fresh, pin.ready)
        })
    };
    // Joined with the release still pending: the binder never waited on it.
    let (fresh, ready) = binder.join().unwrap();
    assert!(!events.done.lock().contains(&release));
    assert!(fresh, "filled again, fresh for the transform");
    assert_eq!(events.unmaps.load(Ordering::Relaxed), 0, "reused in place, never unmapped");
    let fill = *events.fences.lock().first().unwrap();
    assert_eq!(ready, Event(fill));
    assert_eq!(*events.waits.lock(), vec![(release, 0), (fill, 2)], "the copy after the release, the binder after the copy: in-stream");
    gate.open();
    completes.join().unwrap();
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

#[test]
fn a_failed_release_record_never_synchronizes_a_deleter() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "failed_record_child", "--nocapture"]).output().unwrap();
    assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
}

#[test]
#[ignore = "subprocess entry for a_failed_release_record_never_synchronizes_a_deleter"]
fn failed_record_child() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("record-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let pin = mem.bind(id, 0, 7).unwrap();
    events.fail_record.store(true, Ordering::SeqCst);
    drop(pin);
    assert!(events.syncs.lock().is_empty(), "a host callback may need the deleter's GIL or allocator lock");
    assert!(matches!(mem.bind(id, 0, 0), Err(varena::Error::Ownership(_))));
    drop(mem);
    assert_eq!(events.unmaps.load(Ordering::SeqCst), 0, "unknown completion retains the physical mapping");
}

/// A driver-reported OOM is stronger evidence than an unchanged free-byte estimate. There
/// may still be completed idle pages to reclaim; exhausting without trying them strands room.
#[test]
fn an_allocation_shortfall_reclaims_even_when_reported_free_bytes_are_unchanged() {
    for failures in [1, 64] {
        let dir = tempfile::tempdir().unwrap();
        let uuid = format!("hidden-oom-{failures}-{}", std::process::id());
        let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 4 * REGION_ALIGN };
        let _scope = Scope::new(&uuid);
        let events = Arc::new(Events::default());
        let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
        mem.load(stored(dir.path(), &uuid, "idle", 1), Level::Gpu).unwrap();
        mem.settle();
        let target = mem.load(set(dir.path(), &uuid), Level::Ram).unwrap();
        events.hidden_ooms.store(failures, Ordering::SeqCst);
        let creates = events.creates.load(Ordering::SeqCst);
        let result = mem.bind(target, 0, 0);
        if failures == 1 {
            assert!(result.is_ok(), "a completed idle page should rescue this allocation");
        } else {
            assert!(matches!(result, Err(varena::Error::Exhausted(_))), "no-progress retries must end");
        }
        assert!(events.unmaps.load(Ordering::SeqCst) > 0, "the idle page was actually reclaimed");
        assert!(events.creates.load(Ordering::SeqCst) - creates <= 2, "one failed attempt and one retry after real reclaim");
    }
}

/// Python review 1, item 2: a page's create that finds the device fuller than the table (a
/// foreign allocation landed between the decision and the create) is decided again on fresh
/// measurements, never a device error: here the cold page goes and the bind succeeds; with
/// nothing left the bind ends exhausted.
#[test]
fn a_create_short_of_memory_decides_again() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("create-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 4 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let named = |name: &str, regions: u64| {
        let path = dir.path().join(format!("{name}.bin"));
        std::fs::write(&path, vec![7u8; (regions * REGION_ALIGN) as usize]).unwrap();
        let obj = ObjectRef { sha256: format!("obj-{name}"), length: regions * REGION_ALIGN };
        let region = |r: u64| Region { offset: r * REGION_ALIGN, nbytes: REGION_ALIGN, span: REGION_ALIGN, items: vec![Item { offset: r * REGION_ALIGN, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * REGION_ALIGN, len: REGION_ALIGN }), part_end: true }] };
        let layout = Layout {
            parts: (0..regions).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: REGION_ALIGN }).collect(),
            regions: (0..regions).map(region).collect(),
            nbytes: regions * REGION_ALIGN,
            digest: format!("layout-{name}-{uuid}"),
            names: (0..regions).map(|r| vec![format!("t{r}")]).collect(),
        };
        let source = Source::files(vec![(obj.clone(), File::open(&path).unwrap())], false, 0);
        let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", &uuid);
        SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
    };
    let a = mem.load(named("a", 2), Level::Ram).unwrap();
    mem.load(named("n", 1), Level::Gpu).unwrap();
    // 3 pages free in the table's view; at the create, 2.5 land elsewhere.
    events.lands.store(2 * REGION_ALIGN + REGION_ALIGN / 2, Ordering::SeqCst);
    let pin = mem.bind(a, 0, 0).unwrap();
    assert_eq!(mem.classes()[varena::table::Class::N.index()], 0, "the cold page went for it");
    // Everything else lands: nothing is left to give (the first page is still bound), and the
    // bind says so, short by what the page lacks of the measured free half page (S2's
    // v2-native: never "1 byte short" from a stale estimate).
    events.lands.store(4 * REGION_ALIGN, Ordering::SeqCst);
    match mem.bind(a, 1, 0) {
        Err(varena::Error::Exhausted(e)) => assert_eq!(e.need, REGION_ALIGN / 2, "{e:?}"),
        other => panic!("exhausted: {:?}", other.map(|p| p.ptr)),
    }
    drop(pin);
}

/// A stored set `name` of `regions` regions, its fills kept (a declared, stored identity).
fn stored(dir: &std::path::Path, uuid: &str, name: &str, regions: u64) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![7u8; (regions * REGION_ALIGN) as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: regions * REGION_ALIGN };
    let region = |r: u64| Region { offset: r * REGION_ALIGN, nbytes: REGION_ALIGN, span: REGION_ALIGN, items: vec![Item { offset: r * REGION_ALIGN, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * REGION_ALIGN, len: REGION_ALIGN }), part_end: true }] };
    let layout = Layout {
        parts: (0..regions).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: REGION_ALIGN }).collect(),
        regions: (0..regions).map(region).collect(),
        nbytes: regions * REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..regions).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// One Memory per device per process: a row is a process.
#[test]
fn a_device_opens_once_per_process() {
    let g = GpuId { uuid: format!("once-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let first = Memory::open(g.clone(), Box::new(HostBackend::new(g.memory))).unwrap();
    assert!(matches!(Memory::open(g.clone(), Box::new(HostBackend::new(g.memory))), Err(varena::Error::Invalid(_))));
    drop(first);
    drop(Memory::open(g.clone(), Box::new(HostBackend::new(g.memory))).unwrap());
}

/// Review 3, item 15: a bind that needs a page only its own thread pins fails at once (a typed
/// error), never a hang: re-binding a page it is transforming, or switching its fill identity
/// under its own hold.
#[test]
fn a_bind_on_its_own_threads_pin_fails_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("selfpin-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let mem = Memory::open(g.clone(), Box::new(HostBackend::new(g.memory))).unwrap();
    let spec = set(dir.path(), &g.uuid);
    let id = mem.load(spec.clone(), Level::Ram).unwrap();
    let fresh = mem.bind(id, 0, 0).unwrap();
    assert!(fresh.fresh);
    assert!(matches!(mem.bind(id, 0, 0), Err(varena::Error::Held(_))), "it is transforming it");
    fresh.filled();
    let hold = mem.bind(id, 0, 0).unwrap();
    let mut other = spec;
    other.fingerprint.fill = Fill::Declared("lora-b".into());
    mem.load(other, Level::Ram).unwrap();
    assert!(matches!(mem.bind(id, 0, 0), Err(varena::Error::Held(_))), "another identity under its own hold");
    drop((fresh, hold));
    assert!(mem.bind(id, 0, 0).unwrap().fresh, "unheld, it fills again");
}

/// Review 4, NEW-1: the device's VMM calls (which may wait on queued device work) run off the
/// state lock: while an eviction's unmap waits, its binder sleeps on varena's condvar and
/// another thread's calls go through.
#[test]
fn a_waiting_unmap_holds_no_lock() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("unmapwait-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    mem.load(stored(dir.path(), &uuid, "a", 1), Level::Gpu).unwrap();
    let b = mem.load(stored(dir.path(), &uuid, "b", 1), Level::Ram).unwrap();
    // b's page needs a's: a's unmap is held, as on a card with work queued.
    events.hold_unmap.store(true, Ordering::SeqCst);
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || drop(mem.bind(b, 0, 0).unwrap()))
    };
    events.unmap_waits.pass();
    // The unmap waits; the state lock is free.
    mem.classes();
    events.unmap_gate.open();
    binder.join().unwrap();
}

/// The lead's memory6 suspect: a drain queries only each stream's oldest pending event (one
/// stream's events complete in order), so a lock costs a query per stream, not one per page
/// with a release pending.
#[test]
fn a_drain_queries_each_stream_once() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("drain-{}", std::process::id());
    let pages = 64;
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: pages * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let id = mem.load(stored(dir.path(), &uuid, "s", pages), Level::Gpu).unwrap();
    mem.settle();
    // Every page bound and let go on one of two streams: 64 release events pending.
    let first = events.next.load(Ordering::SeqCst) + 1;
    for b in 0..pages as u32 {
        drop(mem.bind(id, b, 1 + b as usize % 2).unwrap());
    }
    let last = events.next.load(Ordering::SeqCst);
    let per_lock = || {
        let before = events.queries.load(Ordering::SeqCst);
        mem.classes();
        events.queries.load(Ordering::SeqCst) - before
    };
    per_lock();
    assert!(per_lock() <= 2, "a query per stream, not per pending event");
    // They complete in order: each lock pops what completed and stops at the next.
    (first..first + 8).for_each(|e| events.complete(e));
    assert!(per_lock() <= 2 + 8);
    (first + 8..=last).for_each(|e| events.complete(e));
    per_lock();
    assert_eq!(per_lock(), 0, "nothing pending");
}

/// ILLEGAL_ADDRESS audit, 2: a fill is issued only after its page is mapped (the mechanism's
/// map, set-access included, has returned): with the map held, nothing writes its address.
#[test]
fn a_fill_is_issued_only_after_its_map() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("mapfirst-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 4 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    events.hold_map.store(true, Ordering::SeqCst);
    let loading = {
        let (mem, dir, uuid) = (mem.clone(), dir.path().to_path_buf(), uuid.clone());
        std::thread::spawn(move || mem.load(stored(&dir, &uuid, "m", 3), Level::Gpu).unwrap())
    };
    events.map_waits.pass();
    assert!(!events.calls.lock().iter().any(|(c, _)| *c == "target"), "a fill was issued while its map waits");
    events.map_gate.open();
    loading.join().unwrap();
    let calls = events.calls.lock().clone();
    for (i, (c, at)) in calls.iter().enumerate() {
        if *c == "target" {
            assert!(calls[..i].iter().any(|(m, a)| *m == "map" && a == at), "a fill of {at:#x} before its map: {calls:?}");
        }
    }
}

#[test]
fn last_public_owner_retires_only_after_the_consumer_completes() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("last-public-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let id = mem.load(stored(dir.path(), &uuid, "s", 1), Level::Ram).unwrap();
    let pin = mem.bind(id, 0, 1).unwrap();
    drop(mem);
    let queries = events.queries.load(Ordering::SeqCst);
    let (sent, dropped) = std::sync::mpsc::channel();
    let dropping = std::thread::spawn(move || { drop(pin); sent.send(()).unwrap(); });
    dropped.recv_timeout(std::time::Duration::from_secs(5)).expect("dropping a DLPack/public pin must not wait for its consumer");
    dropping.join().unwrap();
    let pending = events.next.load(Ordering::SeqCst);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while events.queries.load(Ordering::SeqCst) == queries && std::time::Instant::now() < deadline { std::thread::yield_now(); }
    assert!(events.queries.load(Ordering::SeqCst) > queries, "retirement inspected the pending event");
    assert_eq!(events.unmaps.load(Ordering::SeqCst), 0, "last public drop preserves the consumer's mapping");
    events.complete(pending);
    while events.unmaps.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline { std::thread::yield_now(); }
    assert_eq!(events.unmaps.load(Ordering::SeqCst), 1, "retirement completes without another public API call");
}

/// A set of `regions` regions whose fill is undeclared: never kept, so its idle pages are the
/// allocator callback's to reclaim.
fn undeclared(dir: &std::path::Path, uuid: &str, name: &str, regions: u64) -> SetSpec {
    let mut spec = stored(dir, uuid, name, regions);
    spec.fingerprint = Fingerprint::new(&spec.layout, &[], Fill::Undeclared, "release-1", "0/1", uuid);
    spec
}

/// S2's native old_and_new_fill_identities_at_once (v2-native, run 13376): the fill identity
/// changes while the old range's consumer still runs, and a fresh range's preparation fills
/// the page again in place, after that consumer in-stream (never on the host). Its own claims
/// hold the page for it; they are not pins that keep the old bytes (they made it refuse Held).
#[test]
fn a_range_prepared_after_an_identity_change_fills_its_page_again() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("newidentity-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let t = mem.load(undeclared(dir.path(), &uuid, "t", 1), Level::Ram).unwrap();
    let old = mem.retain(t, &[0]).unwrap();
    drop(old.prepare(1).unwrap());
    let consumer = events.next.load(Ordering::SeqCst);
    mem.fill(t, Fill::Declared("v2".into()));
    assert!(!old.valid(), "the old range is stale");
    let new = mem.retain(t, &[0]).unwrap();
    let prepared = std::thread::scope(|s| {
        let preparing = s.spawn(|| {
            let out = new.prepare(2).map(|pins| pins.iter().all(|p| p.fresh));
            events.log("prepared");
            out
        });
        // Back, or waiting on the host for an event (the old consumer's): then it ends, so the
        // case fails instead of hanging.
        events.await_log(|log| log.contains(&"prepared") || log.contains(&"finish"));
        let waited = !events.kept.lock().contains(&"prepared");
        events.complete(consumer);
        (preparing.join().unwrap(), waited)
    });
    let (fresh, waited) = prepared;
    assert!(matches!(fresh, Ok(true)), "the v2 page is its caller's to bake: {fresh:?}");
    assert!(!waited, "the preparation waited on the host for the old consumer");
    assert!(events.waits.lock().contains(&(consumer, 0)), "the refill waits in-stream for the old consumer");
}


/// S2 (64 evicting binds at 71 ms each on an L4): a bind's work is the pages it touches, never
/// every resident page. Here its driver queries: binds evicting idle pages ask the same number
/// of event queries beside 16 idle pages as beside 48.
#[cfg(target_os = "linux")]
#[test]
fn a_binds_work_does_not_grow_with_the_resident_pages() {
    let queries = |idle: u64| {
        let dir = tempfile::tempdir().unwrap();
        let uuid = format!("bindcost{idle}-{}", std::process::id());
        let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: idle * REGION_ALIGN };
        let _scope = Scope::new(&g.uuid);
        let events = Arc::new(Events::default());
        let mem = Memory::open(g.clone(), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
        let a = mem.load(stored(dir.path(), &uuid, "a", 8), Level::Ram).unwrap();
        mem.load(stored(dir.path(), &uuid, "idle", idle + 1), Level::Gpu).unwrap();
        mem.settle();
        let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
        let pass = r.pass(a, "k").unwrap();
        let before = events.queries.load(Ordering::SeqCst);
        let pins: Vec<_> = (0..8).map(|b| mem.bind(a, b, 0).unwrap()).collect();
        let asked = events.queries.load(Ordering::SeqCst) - before;
        drop((pins, pass));
        asked
    };
    let (few, many) = (queries(16), queries(48));
    assert_eq!(few, many, "8 binds asked {few} event queries beside 16 idle pages, {many} beside 48");
}



