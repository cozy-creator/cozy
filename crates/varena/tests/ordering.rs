//! Ordering against device events: a page whose transform is still running on the bind stream
//! is never made read-only or exported, a pin's release is not complete before its event, and
//! no host thread waits on a device event or on custody. Host pages fill synchronously, so the
//! backend here is the host backend with stream events that complete only when the test says
//! (as a card's do when their work finishes), and with the stream waits it was asked for.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Measured, Memory, Next, Physical, Plan, Result, SetSpec, Step, Torch, Va};

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
    /// Mapping calls by address, in order: map, target (a fill writes), save (a copy out read),
    /// unmap.
    calls: Mutex<Vec<(&'static str, u64)>>,
    /// The next map, fill target or save waits until its gate opens (its `waits` opens then).
    hold_map: AtomicBool,
    map_gate: Gate,
    map_waits: Gate,
    hold_target: AtomicBool,
    target_gate: Gate,
    target_waits: Gate,
    hold_save: AtomicBool,
    /// The next driver free-bytes read waits (under the state lock, as refresh reads it).
    hold_free: AtomicBool,
    free_gate: Gate,
    free_waits: Gate,
    save_gate: Gate,
    save_waits: Gate,
    fail_save: AtomicBool,
    fail_save_drain: AtomicBool,
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

    /// Blocks until the log of calls satisfies `until`.
    fn await_log(&self, until: impl Fn(&[&'static str]) -> bool) {
        let mut log = self.kept.lock();
        while !until(&log) {
            self.logged.wait(&mut log);
        }
    }

    /// Blocks until varena retires `event`.
    fn await_retired(&self, event: u64) {
        let mut retired = self.retired.lock();
        while !retired.contains(&event) {
            self.retiring.wait(&mut retired);
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
    fn close(&self) {
        *self.open.lock() = false;
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
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> Result<()> {
        held(&self.events.hold_save, &self.events.save_waits, &self.events.save_gate);
        self.host.save(va, offset, host, after)?;
        self.events.calls.lock().push(("save", va.base + offset));
        if self.events.fail_save.swap(false, Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "save", code: 999, name: "copy completion unknown".into() });
        }
        Ok(())
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
        let mut done = self.events.done.lock();
        while event != Event::DONE && !done.contains(&event.0) {
            self.events.completed.wait(&mut done);
        }
        Ok(())
    }
    fn register_host(&self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn unregister_host(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn retire(&self, event: Event) {
        self.events.retired.lock().insert(event.0);
        self.events.retiring.notify_all();
    }
    fn drain(&self) -> Result<()> {
        Ok(())
    }
    fn drain_save(&self) -> Result<()> {
        self.events.log("drain_save");
        if self.events.fail_save_drain.load(Ordering::SeqCst) {
            return Err(varena::Error::Device { op: "drain_save", code: 999, name: "stream completion unknown".into() });
        }
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

#[test]
fn a_page_is_kept_only_after_its_transform_completes() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("order-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Memory::open(g.clone(), None, Box::new(backend)).unwrap();
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["lora".into()], passes: 1 }] }).unwrap();
    let pass = r.pass(id, "fwd", "1x1", Torch::default()).unwrap();
    let pin = mem.bind(id, 0, 1).unwrap();
    assert!(pin.fresh, "a declared fill comes back fresh");
    pin.filled();
    let transform = events.next.load(Ordering::Relaxed);
    drop(pin);
    // Decisions run (the lock drains releases), the transform's event still pending.
    for _ in 0..3 {
        mem.torch_limit();
    }
    assert!(events.kept.lock().is_empty(), "kept while its transform ran: {:?}", events.kept.lock());
    // Another binder waits on the transform in-stream and gets the page, not fresh.
    let again = mem.bind(id, 0, 1).unwrap();
    assert!(!again.fresh);
    assert_eq!(again.ready, Event(transform), "the second binder orders after the transform");
    drop(again);
    let last = events.next.load(Ordering::Relaxed);
    events.complete(transform);
    mem.torch_limit();
    // Its consumers' releases are pending: no protect yet (a page in use is never protected).
    assert!(events.kept.lock().is_empty(), "protected while its consumers' streams ran: {:?}", events.kept.lock());
    (1..=last).for_each(|e| events.complete(e));
    mem.torch_limit();
    // The mechanism makes it read-only, then exports it.
    events.await_log(|log| log.len() >= 2);
    assert_eq!(*events.kept.lock(), vec!["protect", "export"], "read-only, then exported, once complete");
    pass.end(Default::default(), Torch::default());
    drop(r);
}

/// Review 1, finding 12: a page released with its transform undone goes, but only after the
/// release event of the stream that last read it completes.
#[test]
fn a_page_unmaps_only_after_its_last_consumer() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("consumer-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Memory::open(g.clone(), None, Box::new(backend)).unwrap();
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["lora".into()], passes: 1 }] }).unwrap();
    let _pass = r.pass(id, "fwd", "1x1", Torch::default()).unwrap();
    let pin = mem.bind(id, 0, 1).unwrap();
    assert!(pin.fresh);
    drop(pin);
    let release = events.next.load(Ordering::Relaxed);
    for _ in 0..3 {
        mem.torch_limit();
    }
    assert_eq!(events.unmaps.load(Ordering::Relaxed), 0, "unmapped while its consumer's stream ran");
    events.complete(release);
    mem.torch_limit();
    while events.unmaps.load(Ordering::Relaxed) == 0 {
        ledger.await_event().unwrap();
    }
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
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["lora".into()], passes: 1 }] }).unwrap();
    let _pass = r.pass(id, "fwd", "1x1", Torch::default()).unwrap();
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
// Every test in this binary uses the same process-wide RAM-tier service and its headroom
// source. Isolate fixtures from one another; concurrency inside a case remains unchanged.
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

/// A custody whose answers wait for `gate`.
fn gated_custody(gpu: &str, gate: Arc<Gate>) {
    serve_custody(gpu, gate, Arc::new(Log::default()));
}

/// The calls a fake custody served, per connection, and its connections (to cut them).
#[derive(Default)]
struct Log {
    calls: Mutex<Vec<(usize, String)>>,
    seen: Condvar,
    streams: Mutex<Vec<std::os::unix::net::UnixStream>>,
    /// The handles it keeps, by (fingerprint key, page), as custody does: they hold the memory.
    kept: Mutex<std::collections::HashMap<(String, u64), Owned>>,
}

/// A call's (fingerprint key, page).
fn kept_key(call: &serde_json::Value) -> Option<(String, u64)> {
    let body = call.as_object()?.values().next()?;
    Some((body.get("fp")?.as_str()?.to_string(), body.get("page")?.as_u64()?))
}

impl Log {
    fn await_call(&self, f: impl Fn(&(usize, String)) -> bool) {
        let mut calls = self.calls.lock();
        while !calls.iter().any(&f) {
            self.seen.wait(&mut calls);
        }
    }

    /// Every connection breaks, as when the holder exits.
    fn cut(&self) {
        self.streams.lock().drain(..).for_each(|s| drop(s.shutdown(std::net::Shutdown::Both)));
    }
}

fn serve_custody(gpu: &str, gate: Arc<Gate>, log: Arc<Log>) {
    let uid = unsafe { libc::geteuid() };
    let addr = SocketAddr::from_abstract_name(format!("cozy-varena-{uid}-test{}-custody-{gpu}", std::process::id())).unwrap();
    let listener = UnixListener::bind_addr(&addr).unwrap();
    std::thread::spawn(move || {
        for (conn, s) in listener.incoming().enumerate() {
            let mut s = s.unwrap();
            log.streams.lock().push(s.try_clone().unwrap());
            let (gate, log) = (gate.clone(), log.clone());
            std::thread::spawn(move || loop {
                let mut head = [0u8; 5];
                if s.read_exact(&mut head).is_err() {
                    return;
                }
                let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
                let mut body = vec![0u8; len];
                s.read_exact(&mut body).unwrap();
                let handle = (head[4] == 1).then(|| platform::handles::recv_fd(&s).unwrap());
                // Received (logged), then served once the gate lets it through.
                let call = String::from_utf8_lossy(&body).to_string();
                log.calls.lock().push((conn, call.clone()));
                log.seen.notify_all();
                gate.pass();
                if let Some(key) = serde_json::from_slice(&body).ok().as_ref().and_then(kept_key) {
                    match handle {
                        Some(h) => drop(log.kept.lock().insert(key, h)),
                        None if call.contains("Withdraw") => drop(log.kept.lock().remove(&key)),
                        None => {}
                    }
                }
                let reply = if call.contains("Import") { r#"{"Imported":{"found":false}}"# } else if call.contains("Release") { r#"{"Released":{"freed":0}}"# } else { r#""Done""# };
                let mut out = (reply.len() as u32).to_le_bytes().to_vec();
                out.push(0);
                out.extend_from_slice(reply.as_bytes());
                if s.write_all(&out).is_err() {
                    return;
                }
            });
        }
    });
}

/// Round 2, finding 1 (the reviewer's deadlock): with custody answering nothing, a binder on a
/// filling page while exports queue, and an unload whose withdraw queues behind them, both
/// return. Exports and withdraws go to one sender thread and no caller waits on them; no thread
/// holds the state lock with another. (A load's import is a question it waits on custody for.)
#[test]
fn no_call_waits_on_custody() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("deadlock-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 64 << 20 };
    let _scope = Scope::new(&g.uuid);
    let custody = Arc::new(Gate::default());
    custody.open();
    gated_custody(&uuid, custody.clone());
    let events = Arc::new(Events::default());
    let backend = Stream { host: HostBackend::new(g.memory), events: events.clone() };
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(backend)).unwrap());
    let named = |name: &str| {
        let mut s = set(dir.path(), &uuid);
        let mut layout = (*s.layout).clone();
        layout.digest = format!("layout-{name}-{uuid}");
        s.name = name.into();
        s.fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", &uuid);
        s.layout = Arc::new(layout);
        s
    };
    let c = mem.load(named("c"), Level::Gpu).unwrap();
    let a = mem.load(named("a"), Level::Ram).unwrap();
    let b = mem.load(named("b"), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["b".into(), "a".into()], passes: 1 }] }).unwrap();
    let _pass = r.pass(b, "fwd", "1x1", Torch::default()).unwrap();
    // Custody stops answering. A's page is filled and kept: its export holds the sender thread.
    custody.close();
    drop(mem.bind(a, 0, 0).unwrap());
    // X binds B, whose fill holds in its fence (a long copy): X waits for the fill.
    events.hold_fence.store(true, Ordering::SeqCst);
    let x = {
        let mem = mem.clone();
        std::thread::spawn(move || drop(mem.bind(b, 0, 0).unwrap()))
    };
    // Meanwhile C is unloaded: its withdraw queues too.
    mem.unload(c);
    events.fence_gate.open();
    x.join().unwrap();
    // Everything returned with custody still answering nothing.
    custody.open();
}

/// Round 2, finding 12: when custody's holder changes (a TensorD restart), every kept page is
/// exported again to the new one.
#[test]
fn kept_pages_go_to_a_new_custody_holder() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("holder-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 64 << 20 };
    let _scope = Scope::new(&g.uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate, log.clone());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: Arc::new(Events::default()) })).unwrap();
    let named = |name: &str| {
        let mut s = set(dir.path(), &uuid);
        let mut layout = (*s.layout).clone();
        layout.digest = format!("layout-{name}-{uuid}");
        s.name = name.into();
        s.fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", &uuid);
        s.layout = Arc::new(layout);
        s
    };
    let a = named("a");
    let key = a.fingerprint.key();
    mem.load(a, Level::Gpu).unwrap();
    log.await_call(|(conn, call)| *conn == 0 && call.contains("Export") && call.contains(&key));
    // The holder goes; the next call reaches a new one.
    log.cut();
    mem.load(named("b"), Level::Ram).unwrap();
    log.await_call(|(conn, call)| *conn > 0 && call.contains("Export") && call.contains(&key));
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
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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
        let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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
    let cold: u64 = ledger.status().rows.iter().map(|r| r.classes[varena::table::Class::N.index()]).sum();
    assert_eq!(cold, 0, "the cold page went for it");
    drop(pin);
    // Everything else lands: nothing is left to give, and the bind says so.
    events.lands.store(4 * REGION_ALIGN, Ordering::SeqCst);
    assert!(matches!(mem.bind(a, 1, 0), Err(varena::Error::Exhausted(_))));
}

/// A card's view of memory: a page's bytes are in use while any handle to them lives, the
/// process's own or custody's (custody runs in this process), and so are bytes allocated outside
/// varena (torch's, or another process's). An allocation needs `slack` bytes free beyond its own
/// (the driver's metadata: a card refused 2 MiB with 3.3 MB free, and 1 GiB with 1.2 MiB over).
struct Card {
    host: HostBackend,
    limit: u64,
    slack: u64,
    /// id → (inode, bytes, destroyed by this process).
    pages: Mutex<std::collections::HashMap<u64, (u64, u64, bool)>>,
    /// This process's allocations outside varena (torch's), and another process's bytes.
    torch: AtomicU64,
    foreign: AtomicU64,
    /// The next unmap waits (as a card's does on queued work) until the gate opens.
    hold_unmap: AtomicBool,
    unmap_gate: Gate,
    unmap_waits: Gate,
}

fn inode(h: &Owned) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::File::from(h.try_clone().unwrap()).metadata().unwrap().ino()
}

impl Card {
    fn new(limit: u64, slack: u64) -> Arc<Card> {
        Arc::new(Card { host: HostBackend::new(limit), limit, slack, pages: Mutex::default(), torch: AtomicU64::default(), foreign: AtomicU64::default(), hold_unmap: AtomicBool::default(), unmap_gate: Gate::default(), unmap_waits: Gate::default() })
    }

    fn used(&self) -> u64 {
        let open: HashSet<u64> = std::fs::read_dir("/proc/self/fd").unwrap().flatten().filter_map(|e| std::fs::metadata(e.path()).ok()).map(|m| std::os::unix::fs::MetadataExt::ino(&m)).collect();
        let pages: u64 = self.pages.lock().values().filter(|(ino, _, gone)| !gone || open.contains(ino)).map(|(_, len, _)| len).sum();
        pages + self.torch.load(Ordering::SeqCst) + self.foreign.load(Ordering::SeqCst)
    }

    fn free(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }

    fn fits(&self, len: u64) -> bool {
        self.used() + len + self.slack <= self.limit
    }

    /// An allocation of torch's, as a card serves it: refused without room.
    fn take(&self, len: u64) -> bool {
        let fits = self.fits(len);
        if fits {
            self.torch.fetch_add(len, Ordering::SeqCst);
        }
        fits
    }

    fn give(&self, len: u64) {
        self.torch.fetch_sub(len, Ordering::SeqCst);
    }
}

struct Driver(Arc<Card>);

impl Backend for Driver {
    fn capacity(&self) -> u64 {
        self.0.limit
    }
    fn free(&self) -> Result<u64> {
        Ok(self.0.free())
    }
    fn process_used(&self) -> Result<u64> {
        Ok(self.0.host.process_used()? + self.0.torch.load(Ordering::SeqCst))
    }
    fn opened_context(&self) -> u64 {
        0
    }
    fn granularity(&self) -> u64 {
        self.0.host.granularity()
    }
    fn reserve(&self, len: u64) -> Result<Va> {
        self.0.host.reserve(len)
    }
    fn release(&self, va: Va) {
        self.0.host.release(va)
    }
    fn create(&self, len: u64) -> Result<Physical> {
        if !self.0.fits(len) {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        let page = self.0.host.create(len)?;
        let ino = inode(&self.0.host.export(&page)?);
        self.0.pages.lock().insert(page.id, (ino, len, false));
        Ok(page)
    }
    fn destroy(&self, page: Physical) -> Result<()> {
        if let Some(p) = self.0.pages.lock().get_mut(&page.id) {
            p.2 = true;
        }
        self.0.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        self.0.host.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        self.0.host.protect(va, offset, len, access)
    }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        held(&self.0.hold_unmap, &self.0.unmap_waits, &self.0.unmap_gate);
        self.0.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> {
        self.0.host.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        self.0.host.import(handle, len)
    }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
        self.0.host.target(va, offset, len)
    }
    fn save(&self, va: Va, offset: u64, host: &mut [u8], after: Event) -> Result<()> {
        self.0.host.save(va, offset, host, after)
    }
    fn fence(&self) -> Result<Event> {
        self.0.host.fence()
    }
    fn record(&self, stream: usize) -> Result<Event> {
        self.0.host.record(stream)
    }
    fn query(&self, event: Event) -> Result<bool> {
        self.0.host.query(event)
    }
    fn elapsed(&self, from: Event, to: Event) -> Result<Option<u64>> {
        self.0.host.elapsed(from, to)
    }
    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        self.0.host.wait(event, stream)
    }
    fn synchronize(&self, stream: usize) -> Result<()> {
        self.0.host.synchronize(stream)
    }
    fn finish(&self, event: Event) -> Result<()> {
        self.0.host.finish(event)
    }
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> {
        self.0.host.register_host(ptr, len)
    }
    fn unregister_host(&self, ptr: u64) -> Result<()> {
        self.0.host.unregister_host(ptr)
    }
    fn retire(&self, event: Event) {
        self.0.host.retire(event)
    }
    fn drain(&self) -> Result<()> {
        self.0.host.drain()
    }
    fn drain_save(&self) -> Result<()> {
        self.0.host.drain_save()
    }
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

/// Review 3, N1: an evicted kept page's memory is custody's until its Withdraw is sent; the
/// demand that evicted it waits for that before mapping into the bytes, so one eviction is
/// enough and the bind succeeds.
#[test]
fn an_evicted_kept_page_is_free_once_withdrawn() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("withdraw-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 3 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate, log.clone());
    let mem = Memory::open(g.clone(), None, Box::new(Driver(Card::new(g.memory, 0)))).unwrap();
    let k = stored(dir.path(), &uuid, "k", 2);
    let key = k.fingerprint.key();
    mem.load(k, Level::Gpu).unwrap();
    for page in 0..2u64 {
        log.await_call(|(_, call)| call.contains("Export") && call.contains(&key) && call.contains(&format!("\"page\":{page}")));
    }
    let a = mem.load(stored(dir.path(), &uuid, "a", 2), Level::Ram).unwrap();
    let pins = [mem.bind(a, 0, 0).unwrap(), mem.bind(a, 1, 0).unwrap()];
    let withdraws = log.calls.lock().iter().filter(|(_, c)| c.contains("Withdraw")).count();
    assert_eq!(withdraws, 1, "one kept page made room, once custody let it go");
    assert_eq!(log.kept.lock().keys().filter(|(fp, _)| *fp == key).count(), 1, "the other stays kept");
    drop(pins);
}

/// Review 3, item 6: where the process's own bytes cannot be measured (WDDM, a foreign pid
/// namespace), its context is the one open measured.
#[test]
fn an_unmeasurable_process_keeps_its_opened_context() {
    let g = GpuId { uuid: format!("opened-{}", std::process::id()), model: "host".into(), memory: 64 << 20 };
    let _scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    events.unmeasured.store(5 << 20, Ordering::SeqCst);
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events })).unwrap();
    mem.torch_limit();
    let pid = std::process::id();
    assert_eq!(ledger.status().rows.iter().find(|r| r.pid == pid && r.context > 0).map(|r| r.context), Some(5 << 20));
    drop(mem);
}

/// Review 3, N6 / review 4, NEW-1, NEW-2: a kept page whose fill identity changed is never
/// written again: a fresh page takes its place (no re-protect), so custody's copy under the old
/// key keeps the old bytes whatever happens (a kill -9 at any point imports old bytes or
/// nothing), and the refill never waits on custody.
#[cfg(target_os = "linux")]
#[test]
fn a_kept_page_is_never_overwritten_under_its_old_key() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("rekey-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate.clone(), log.clone());
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    let spec = set(dir.path(), &uuid);
    let old = spec.fingerprint.key();
    let id = mem.load(spec.clone(), Level::Ram).unwrap();
    // Baked under lora-a, and kept once the bake completed.
    let pin = mem.bind(id, 0, 1).unwrap();
    // SAFETY: a fresh host page bound to this test alone.
    unsafe { std::ptr::write_bytes(pin.ptr as *mut u8, 0xab, pin.len as usize) };
    let len = pin.len as usize;
    pin.filled();
    // Kept once the bake completed and its consumer let go (a page in use is never protected).
    drop(pin);
    (1..=events.next.load(Ordering::Relaxed)).for_each(|e| events.complete(e));
    mem.torch_limit();
    log.await_call(|(_, c)| c.contains("Export") && c.contains(&old));
    // Another consumer reads it and lets go; custody stops answering; lora-b is loaded.
    drop(mem.bind(id, 0, 1).unwrap());
    gate.close();
    let mut b = spec;
    b.fingerprint.fill = Fill::Declared("lora-b".into());
    mem.load(b, Level::Ram).unwrap();
    // The renewal waits for the consumer's work to finish (review 5, R3).
    events.complete(events.next.load(Ordering::Relaxed));
    let fresh = mem.bind(id, 0, 2).unwrap().fresh;
    assert!(fresh, "refilled for lora-b while custody answered nothing");
    // Custody still holds the old key (its Withdraw waits at the gate), over the old bytes.
    let kept = log.kept.lock().get(&(old.clone(), 0)).map(|h| h.try_clone().unwrap()).expect("custody still keeps the old key");
    // SAFETY: a read-only shared mapping of custody's handle, unmapped below.
    let view = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, std::os::fd::AsRawFd::as_raw_fd(&kept), 0) };
    assert_ne!(view, libc::MAP_FAILED);
    // SAFETY: the mapping above, `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(view as *const u8, len) };
    assert!(bytes.iter().all(|b| *b == 0xab), "custody's copy under the old key was overwritten");
    // SAFETY: our mapping.
    unsafe { libc::munmap(view, len) };
    gate.open();
}

/// S3 gpu5: a capped table beside other occupants of the card. Their bytes take from the card,
/// not from the cap: the table keeps its whole cap while the card has room outside it.
#[test]
fn a_capped_table_keeps_its_cap_beside_other_occupants() {
    let g = GpuId { uuid: format!("capped-{}", std::process::id()), model: "host".into(), memory: 64 << 20 };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    // 40 MB of the card are others' (no row accounts for them).
    events.foreign.store(40 << 20, Ordering::SeqCst);
    let grant = varena::Grant::open(&g, Some(8 << 20)).unwrap();
    let mem = Memory::open(g.clone(), Some(grant), Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    assert_eq!(mem.torch_limit(), 8 << 20, "the whole cap: the card has 24 MB outside the others");
    // The others grow to leave the card less than the cap.
    events.foreign.store(60 << 20, Ordering::SeqCst);
    assert_eq!(mem.torch_limit(), 4 << 20, "min(cap, card - others)");
}

/// One Memory per device per process: a row is a process.
#[test]
fn a_device_opens_once_per_process() {
    let g = GpuId { uuid: format!("once-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let first = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    assert!(matches!(Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))), Err(varena::Error::Invalid(_))));
    drop(first);
    drop(Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap());
}

/// Review 3, item 15: a bind that needs a page only its own thread pins fails at once (a typed
/// error), never a hang: re-binding a page it is transforming, or switching its fill identity
/// under its own hold.
#[test]
fn a_bind_on_its_own_threads_pin_fails_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("selfpin-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
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
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
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
    mem.torch_limit();
    events.unmap_gate.open();
    binder.join().unwrap();
}

/// Review 5, R1 and R3 (the reviewer's probe, event-driven): a Keep queued behind slow
/// mechanism work never protects the page a refill hands out. A page with a Keep queued is
/// renewed (a fresh page, a new life) and the stale Keep is dropped; the renewal unmaps the old
/// page only once its consumers' events complete.
#[test]
fn a_page_with_a_keep_queued_is_renewed_after_its_consumers() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("stalekeep-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 64 << 20 };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    let a = mem.load(stored(dir.path(), &uuid, "a", 1), Level::Gpu).unwrap();
    mem.settle();
    // P's fill lands as an event still pending, so its Keep waits for it.
    events.pending_fences.store(true, Ordering::SeqCst);
    let p = mem.load(stored(dir.path(), &uuid, "p", 1), Level::Gpu).unwrap();
    let filled = *events.fences.lock().last().unwrap();
    // A consumer binds P and lets go: its release event is pending.
    drop(mem.bind(p, 0, 1).unwrap());
    let release = events.next.load(Ordering::SeqCst);
    // The mechanism thread waits in an unmap, as a card's does behind queued work.
    events.hold_unmap.store(true, Ordering::SeqCst);
    mem.unload(a);
    events.unmap_waits.pass();
    // P's fill completes: the next decision queues its Keep behind the held unmap.
    events.complete(filled);
    mem.torch_limit();
    let protects = || events.kept.lock().iter().filter(|c| **c == "protect").count();
    let before = protects();
    // A fill-identity switch (a LoRA change), then a bind: P is renewed, not refilled in place.
    mem.fill(p, Fill::Declared("lora-b".into()));
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || mem.bind(p, 0, 2).unwrap())
    };
    // The bind decided the refill (it gave up P's old ready event) before the Keep runs.
    events.await_retired(filled);
    events.unmap_gate.open();
    // The mechanism runs P's queued Keep, then the renewal, which waits for P's consumer.
    events.await_log(|log| log.contains(&"finish") || log.iter().filter(|c| **c == "protect").count() > before);
    assert_eq!(protects(), before, "the stale Keep protected nothing");
    let unmaps = events.unmaps.load(Ordering::SeqCst);
    events.complete(release);
    let pin = binder.join().unwrap();
    assert!(events.unmaps.load(Ordering::SeqCst) > unmaps, "the old page unmapped after its consumer");
    assert!(pin.fresh);
    // The caller bakes into its fresh page.
    // SAFETY: a fresh host page bound to this test alone.
    unsafe { std::ptr::write_bytes(pin.ptr as *mut u8, 0xcd, pin.len as usize) };
    pin.filled();
}

/// An OOM's Retry comes once its unmaps are done (the mechanism's, off the lock): the caller
/// retries its allocation into bytes the device has freed. The mechanism never waits on the
/// caller's pins.
#[test]
fn an_oom_retries_once_its_unmaps_are_done() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("oomunmap-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 2 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    mem.load(stored(dir.path(), &uuid, "n", 1), Level::Gpu).unwrap();
    mem.load(stored(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    mem.settle();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
    let before = events.unmaps.load(Ordering::SeqCst);
    events.hold_unmap.store(true, Ordering::SeqCst);
    let oom = std::thread::spawn({
        let events = events.clone();
        move || {
            let next = r.oom(None, Some(2 * REGION_ALIGN), None, Torch::default());
            (matches!(next, varena::Next::Retry), events.unmaps.load(Ordering::SeqCst))
        }
    });
    events.unmap_waits.pass();
    events.unmap_gate.open();
    let (retry, seen) = oom.join().unwrap();
    assert!(retry);
    assert!(seen > before, "the Retry came before the cold page's unmap was done");
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
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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
        mem.torch_limit();
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

/// ILLEGAL_ADDRESS audit, 1: no page is unmapped while a fill into it is in flight. A cold
/// load's prefetch is held mid-fill (its copy not yet issued) while an OOM wants every byte:
/// the in-flight page is P, so the OOM takes the rest, and the fill lands in mapped memory
/// (an unmapped page is PROT_NONE here: a write would fault the test).
#[test]
fn a_page_in_flight_is_never_unmapped_for_an_oom() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("inflight-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 3 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    mem.load(stored(dir.path(), &uuid, "n", 1), Level::Gpu).unwrap();
    let a = mem.load(stored(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    mem.settle();
    events.hold_target.store(true, Ordering::SeqCst);
    let loading = {
        let (mem, dir, uuid) = (mem.clone(), dir.path().to_path_buf(), uuid.clone());
        std::thread::spawn(move || mem.load(stored(&dir, &uuid, "p", 1), Level::Gpu).unwrap())
    };
    events.target_waits.pass();
    let filling = events.calls.lock().iter().rev().find(|(c, _)| *c == "target").map(|(_, at)| *at).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
    let pass = r.pass(a, "fwd", "1x1", Torch::default()).unwrap();
    let next = r.oom(Some(a), Some(3 * REGION_ALIGN), None, Torch::default());
    assert!(!events.calls.lock().iter().any(|(c, at)| *c == "unmap" && *at == filling), "the page in flight was unmapped ({next:?})");
    events.target_gate.open();
    let p = loading.join().unwrap();
    drop(pass);
    drop(r);
    let pin = mem.bind(p, 0, 0).unwrap();
    // SAFETY: a mapped page bound to this test; read only.
    assert!(unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) }.iter().all(|b| *b == 7));
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
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
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

/// ILLEGAL_ADDRESS audit, 3: no page is unmapped while the RAM tier's copy reads it. The copy
/// is held mid-way while a bind needs the page's bytes: the bind waits for the copy, then
/// evicts; the unmap comes after the copy.
#[test]
fn a_page_being_saved_is_unmapped_only_after_its_copy() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("saving-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    events.hold_save.store(true, Ordering::SeqCst);
    mem.load(stored(dir.path(), &uuid, "s", 1), Level::Gpu).unwrap();
    events.save_waits.pass();
    let saved = events.calls.lock().iter().find(|(c, _)| *c == "map").map(|(_, at)| *at).unwrap();
    let t = mem.load(stored(dir.path(), &uuid, "t", 1), Level::Ram).unwrap();
    let binding = {
        let mem = mem.clone();
        std::thread::spawn(move || drop(mem.bind(t, 0, 0).unwrap()))
    };
    events.save_gate.open();
    binding.join().unwrap();
    let calls = events.calls.lock().clone();
    let at = |what: &str| calls.iter().position(|(c, a)| *c == what && *a == saved);
    assert!(at("save") < at("unmap") && at("save").is_some(), "unmapped under its copy: {calls:?}");
}

/// An unknown copy completion deliberately retains owners until process exit. Run each
/// case in its own process, so its quarantined tier reservation cannot affect other tests.
#[test]
fn failed_saves_release_memory_only_after_the_save_stream_drains() {
    for drained in [true, false] {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["failed_save_child", "--exact", "--ignored", "--nocapture"])
            .env("VARENA_TEST_SAVE_DRAINED", if drained { "1" } else { "0" })
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "drained={drained}: {}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
#[ignore = "subprocess entry for failed_saves_release_memory_only_after_the_save_stream_drains"]
fn failed_save_child() {
    let drained = std::env::var("VARENA_TEST_SAVE_DRAINED").unwrap() == "1";
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("failedsave-{}", std::process::id());
    let g = GpuId {
        uuid: uuid.clone(),
        model: "host".into(),
        memory: REGION_ALIGN,
    };
    let _scope = Scope::new(&uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(
        g.clone(),
        None,
        Box::new(Stream {
            host: HostBackend::new(g.memory),
            events: events.clone(),
        }),
    )
    .unwrap();
    events.hold_save.store(true, Ordering::SeqCst);
    events.fail_save.store(true, Ordering::SeqCst);
    events.fail_save_drain.store(!drained, Ordering::SeqCst);
    let id = mem
        .load(stored(dir.path(), &uuid, "s", 1), Level::Gpu)
        .unwrap();
    events.save_waits.pass();
    let request = mem.request("save-error", Plan::default()).unwrap();
    events.save_gate.open();
    events.await_log(|calls| calls.contains(&"drain_save"));
    if drained {
        mem.settle();
        request.check().unwrap();
        mem.unload(id);
        mem.settle();
        assert_eq!(
            events.unmaps.load(Ordering::SeqCst),
            1,
            "drained source can be unmapped"
        );
    } else {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while request.check().is_ok() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(matches!(request.check(), Err(varena::Error::Ownership(_))));
        assert!(matches!(
            mem.bind(id, 0, 0),
            Err(varena::Error::Ownership(_))
        ));
        drop(request);
        drop(mem);
        assert_eq!(
            events.unmaps.load(Ordering::SeqCst),
            0,
            "uncertain source survives the last public owner"
        );
    }
}

/// A previous unload can still be waiting for custody when an independent bind starts.
/// Preparation waits outside the state lock; allocator recovery refuses without waiting.
#[test]
fn an_earlier_withdrawal_stays_charged_until_custody_replies() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("pendingwithdraw-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate.clone(), log.clone());
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events })).unwrap());
    let a = mem.load(stored(dir.path(), &uuid, "old", 1), Level::Gpu).unwrap();
    let b = mem.load(stored(dir.path(), &uuid, "next", 1), Level::Ram).unwrap();
    mem.settle();
    gate.close();
    mem.unload(a);
    log.await_call(|(_, call)| call.contains("Withdraw"));
    assert_eq!(mem.torch_limit(), 0, "the queued custody handle still holds the page");
    let request = mem.request("pending", Plan::default()).unwrap();
    assert_eq!(request.reclaim_allocation(None, REGION_ALIGN, Torch::default()).unwrap(), 0, "the allocator never waits for custody");
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || drop(mem.bind(b, 0, 0).unwrap()))
    };
    // Probe the state lock while preparation may be settling, then release the exact hold.
    mem.torch_limit();
    gate.open();
    binder.join().unwrap();
}

#[test]
fn failed_renewal_completion_keeps_the_old_mapping_alive() {
    for mode in ["finish", "unmap", "destroy"] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "failed_renewal_child", "--nocapture"])
            .env("VARENA_TEST_RENEWAL_FAILURE", mode).output().unwrap();
        assert!(output.status.success(), "{mode}: {}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
#[ignore = "subprocess entry for failed_renewal_completion_keeps_the_old_mapping_alive"]
fn failed_renewal_child() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("failedrenewal-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate, log.clone());
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let mut spec = set(dir.path(), &uuid);
    let old = spec.fingerprint.key();
    let id = mem.load(spec.clone(), Level::Ram).unwrap();
    let pin = mem.bind(id, 0, 1).unwrap();
    let ptr = pin.ptr;
    pin.filled();
    // Kept once the bake completed and its consumer let go (a page in use is never protected).
    drop(pin);
    (1..=events.next.load(Ordering::Relaxed)).for_each(|e| events.complete(e));
    mem.torch_limit();
    log.await_call(|(_, c)| c.contains("Export") && c.contains(&old));
    // Another consumer reads it and lets go: its release is pending at the renewal.
    drop(mem.bind(id, 0, 1).unwrap());
    let pending = events.next.load(Ordering::Relaxed);
    spec.fingerprint.fill = Fill::Declared("lora-b".into());
    mem.load(spec, Level::Ram).unwrap();
    let mode = std::env::var("VARENA_TEST_RENEWAL_FAILURE").unwrap();
    let finish = mode == "finish";
    if finish { events.fail_finish.store(true, Ordering::SeqCst); }
    else {
        if mode == "unmap" { events.fail_unmap.store(true, Ordering::SeqCst); }
        else { events.fail_destroy.store(true, Ordering::SeqCst); }
        events.complete(pending);
    }
    assert!(matches!(mem.bind(id, 0, 2), Err(varena::Error::Ownership(_))), "failed release must poison preparation");
    drop(mem);
    assert_eq!(events.unmaps.load(Ordering::SeqCst), u64::from(mode == "destroy"), "only proven-safe unmapping may proceed");
    if finish { assert!(!events.retired.lock().contains(&pending), "uncertain event stays owned"); }
    // SAFETY: the fault path must retain the original mapping through executor exit.
    if mode != "destroy" { assert_eq!(unsafe { *(ptr as *const u8) }, 7); }
}

#[test]
fn last_public_owner_retires_only_after_the_consumer_completes() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("last-public-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 << 20 };
    let _scope = Scope::new(&uuid);
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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

/// Lead's ruling 2 (S3 review / run 13090): the allocator callback never waits for the state
/// lock, and a held lock never makes it refuse: its limit is the last publication's, and it
/// reclaims idle pages offered to it beforehand.
#[test]
fn the_allocator_callback_works_while_the_state_is_locked() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("lockfree-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 4 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    mem.load(undeclared(dir.path(), &uuid, "idle", 2), Level::Gpu).unwrap();
    let a = mem.load(stored(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    mem.settle();
    // Between passes (an unmeasured pass would take the idle pages itself, before it runs).
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
    // Another thread holds the state lock (in a driver read under it).
    events.hold_free.store(true, Ordering::SeqCst);
    let holder = {
        let mem = mem.clone();
        std::thread::spawn(move || mem.torch_limit())
    };
    events.free_waits.pass();
    let limit = mem.allocator_limit();
    let reclaimed = r.reclaim_allocation(None, 2 * REGION_ALIGN, Torch::default());
    let after = mem.allocator_limit();
    // The lock goes back before any assertion, so a failure never hangs the case.
    events.free_gate.open();
    holder.join().unwrap();
    let (limit, reclaimed) = (limit.expect("a limit with the state locked"), reclaimed.unwrap());
    assert_eq!(reclaimed, 2 * REGION_ALIGN, "the idle pages, reclaimed with the state locked");
    assert_eq!(after, Some(limit + reclaimed), "the limit counts them at once");
    drop(r);
    let _ = a;
    // Reconciled: the reclaimed pages are gone from the table, and bind again from the store.
    let idle = mem.load(undeclared(dir.path(), &uuid, "idle", 2), Level::Ram).unwrap();
    drop(mem.bind(idle, 0, 0).unwrap());
}

/// Lead's ruling 1 (S3 review / run 13090, host version): an unmeasured pass on a card full
/// of an idle kept set takes the idle pages before it runs (preparation, outside the
/// allocator), so its allocations find the room; in the callback a kept page is never
/// reclaimable (custody holds its memory).
#[test]
fn an_unmeasured_pass_takes_idle_kept_pages_before_it_runs() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("unmeasured-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 4 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    mem.load(stored(dir.path(), &uuid, "kept", 3), Level::Gpu).unwrap();
    let a = mem.load(stored(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    mem.settle();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
    // In the callback, kept pages are custody's: nothing to reclaim.
    assert_eq!(r.reclaim_allocation(Some(a), REGION_ALIGN, Torch::default()).unwrap(), 0, "kept pages are not the callback's");
    let pass = r.pass(a, "fwd", "1x1", Torch::default()).unwrap();
    mem.settle();
    let n: u64 = ledger.status().rows.iter().map(|r| r.classes[varena::table::Class::N.index()]).sum();
    assert_eq!(n, 0, "the idle kept pages went before the unmeasured pass ran");
    assert!(mem.allocator_limit().unwrap() >= 2 * REGION_ALIGN, "its allocations have the room: {:?}", mem.allocator_limit());
    drop(pass);
}

/// gpu12's ILLEGAL_ADDRESS: a Keep made a page read-only (`cuMemSetAccess`) while kernels of
/// the block it had just been bound to read it, and the read faulted. A Keep protects only idle
/// pages (unpinned, no release pending), each contiguous run of a set in one call; a page in use
/// stays due (Stats' keep_pending) and is kept at its first idle moment; a bind that meets a
/// protect in flight gets its pin only after it.
#[test]
fn a_keep_protects_only_idle_pages_and_a_bind_waits_for_its_protect() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("keepidle-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    let protects = || events.kept.lock().iter().filter(|c| **c == "protect").count();
    let exports = || events.kept.lock().iter().filter(|c| **c == "export").count();

    // Three contiguous pages filled while pinned (their fills' events pending until the pins
    // are taken), then released: due, but in use until their release events complete; then
    // idle together, and kept in one protect.
    let run = mem.load(stored(dir.path(), &uuid, "run", 3), Level::Ram).unwrap();
    events.pending_fences.store(true, Ordering::SeqCst);
    let pins: Vec<_> = (0..3).map(|b| mem.bind(run, b, 1).unwrap()).collect();
    events.pending_fences.store(false, Ordering::SeqCst);
    let fills = events.next.load(Ordering::Relaxed);
    (1..=fills).for_each(|e| events.complete(e));
    mem.torch_limit();
    drop(pins);
    let last = events.next.load(Ordering::Relaxed);
    mem.torch_limit();
    assert_eq!(protects(), 0, "protected while in use");
    (fills + 1..=last).for_each(|e| events.complete(e));
    mem.torch_limit();
    events.await_log(|log| log.iter().filter(|c| **c == "export").count() == 3);
    assert_eq!(*events.protected.lock(), vec![3 * REGION_ALIGN], "one protect for the run");

    // A page bound while its fill's event is pending, held through the fill's completion and
    // released with its release event pending: never protected while in use.
    let r = mem.request("r", Plan::default()).unwrap();
    events.pending_fences.store(true, Ordering::SeqCst);
    let used = mem.load(stored(dir.path(), &uuid, "used", 1), Level::Ram).unwrap();
    let pin = mem.bind(used, 0, 1).unwrap();
    events.pending_fences.store(false, Ordering::SeqCst);
    let fill = *events.fences.lock().last().unwrap();
    events.complete(fill);
    mem.torch_limit();
    assert_eq!(protects(), 1, "protected while pinned");
    assert_eq!(r.stats().keep_pending, REGION_ALIGN, "due, waiting for the page to be idle");
    drop(pin);
    let release = events.next.load(Ordering::Relaxed);
    mem.torch_limit();
    assert_eq!(protects(), 1, "protected while its consumer's stream ran");
    events.complete(release);
    mem.torch_limit();
    events.await_log(|log| log.iter().filter(|c| **c == "export").count() == 4);
    assert_eq!(protects(), 2);
    mem.settle();
    assert_eq!(r.stats().keep_pending, 0, "kept at its first idle moment");

    // A bind that meets a protect in flight waits for it.
    let raced = mem.load(stored(dir.path(), &uuid, "raced", 1), Level::Ram).unwrap();
    events.pending_fences.store(true, Ordering::SeqCst);
    drop(mem.bind(raced, 0, 1).unwrap());
    events.pending_fences.store(false, Ordering::SeqCst);
    events.hold_protect.store(true, Ordering::SeqCst);
    (1..=events.next.load(Ordering::Relaxed)).for_each(|e| events.complete(e));
    mem.torch_limit();
    events.protect_waits.pass();
    let (tid_tx, tid_rx) = std::sync::mpsc::channel();
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || {
            // SAFETY: gettid has no preconditions.
            tid_tx.send(unsafe { libc::gettid() }).unwrap();
            drop(mem.bind(raced, 0, 7).unwrap());
        })
    };
    let tid = tid_rx.recv().unwrap();
    // The protect goes on once the binder returned (a pin under the protect) or sleeps on it.
    let sleeping = || std::fs::read_to_string(format!("/proc/self/task/{tid}/stat")).is_ok_and(|s| s.rsplit(')').next().is_some_and(|rest| rest.trim_start().starts_with('S')));
    while !binder.is_finished() && !sleeping() {
        std::thread::yield_now();
    }
    events.protect_gate.open();
    binder.join().unwrap();
    assert_eq!(events.bound_while_protecting.load(Ordering::SeqCst), 0, "a pin handed out while its page's protect ran");
    assert_eq!(exports(), 5);
    drop(r);
}

/// Native an_oom_in_a_pass / a_measured_pass (graph-core runs 519 and e199) on a host card, after
/// the lead's ruling 1: the card crowded and the rest full of an idle kept set. An unmeasured pass
/// takes the idle pages before it runs, with custody's handles and its own unmaps gone, so its
/// activation fits at once; the next run's measured pass makes its room before it runs too.
/// Neither meets an OOM.
#[test]
fn a_pass_on_a_card_full_of_idle_kept_pages_has_its_room_before_it_runs() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("fullkept-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 24 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let card = Card::new(g.memory, 0);
    let mem = Memory::open(g.clone(), None, Box::new(Driver(card.clone()))).unwrap();
    let m = mem.load(stored(dir.path(), &uuid, "m", 2), Level::Ram).unwrap();
    card.foreign.store(8 * REGION_ALIGN, Ordering::SeqCst);
    let activation = 8 * REGION_ALIGN;
    let plan = Plan { steps: vec![Step { sets: vec!["m".into()], passes: 1 }] };
    for (name, measured) in [("first", false), ("second", true)] {
        mem.load(stored(dir.path(), &uuid, "idle", 17), Level::Gpu).unwrap();
        mem.settle();
        assert!(!card.fits(REGION_ALIGN), "{name}: the idle set fills the card: {} free", card.free());
        let r = mem.request(name, plan.clone()).unwrap();
        let pass = r.pass(m, "main", "1x8", Torch::default()).unwrap();
        let pins: Vec<varena::Pin> = (0..2).map(|b| mem.bind(m, b, 0).unwrap()).collect();
        assert!(card.take(activation), "{name} (measured: {measured}): the pass made its room before it ran: {} free", card.free());
        pass.end(Measured { allocated_peak: activation, slack: 0, sites: vec![] }, Torch { allocated: activation, reserved: activation });
        drop(pins);
        card.give(activation);
        let stats = r.end();
        assert_eq!((stats.retries, stats.leftover), (0, 0), "{name}: {stats:?}");
        assert!(stats.evicted.iter().sum::<u64>() >= activation, "{name}: {stats:?}");
    }
}

/// Lead's ruling 1, the pass side: the room a pass start makes is free on the device when the
/// pass runs (torch allocates outside the mechanism's order, and the allocator callback cannot
/// wait). The idle kept pages an unmeasured pass takes are free once custody lets them go and
/// their unmaps are done: both are held here, and the pass returns only after them.
#[test]
fn a_pass_returns_only_once_the_room_it_made_is_free() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("roomfree-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate.clone(), log.clone());
    let card = Card::new(g.memory, 0);
    let mem = Memory::open(g.clone(), None, Box::new(Driver(card.clone()))).unwrap();
    let k = stored(dir.path(), &uuid, "k", 6);
    let key = k.fingerprint.key();
    mem.load(k, Level::Gpu).unwrap();
    mem.settle();
    assert_eq!(log.kept.lock().keys().filter(|(fp, _)| *fp == key).count(), 6, "the idle set is kept");
    let a = mem.load(stored(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["a".into()], passes: 1 }] }).unwrap();
    // Custody answers nothing, and the next unmap waits.
    gate.close();
    card.hold_unmap.store(true, Ordering::SeqCst);
    let (tid_tx, tid_rx) = std::sync::mpsc::channel();
    std::thread::scope(|t| {
        let passing = t.spawn(|| {
            // SAFETY: gettid has no preconditions.
            tid_tx.send(unsafe { libc::gettid() }).unwrap();
            let pass = r.pass(a, "fwd", "1x1", Torch::default()).unwrap();
            (pass, card.free())
        });
        let tid = tid_rx.recv().unwrap();
        // Its decision is out: an unmap is held and custody has a Withdraw.
        card.unmap_waits.pass();
        log.await_call(|(_, call)| call.contains("Withdraw"));
        let sleeping = || std::fs::read_to_string(format!("/proc/self/task/{tid}/stat")).is_ok_and(|s| s.rsplit(')').next().is_some_and(|rest| rest.trim_start().starts_with('S')));
        while !passing.is_finished() && !sleeping() {
            std::thread::yield_now();
        }
        gate.open();
        card.unmap_gate.open();
        let (pass, free) = passing.join().unwrap();
        assert_eq!(free, 8 * REGION_ALIGN, "the pass returned before the room it made was free");
        drop(pass);
    });
}

/// An OOM in a measured pass on a card full of idle kept pages (native an_oom_in_a_pass): the
/// pass made only its measured room, the activation outgrew it, and each OOM answers Retry with
/// the idle pages; a retry the driver still refuses (its slack, as e199's refused a claim of
/// exactly 1 GiB with 1.19 MiB more reported free) evicts more, and fits.
#[test]
fn an_oom_in_a_measured_pass_takes_idle_kept_pages_and_its_retry_fits() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("oomkept-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 24 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let card = Card::new(g.memory, 3 * REGION_ALIGN / 2);
    let mem = Memory::open(g.clone(), None, Box::new(Driver(card.clone()))).unwrap();
    let m = mem.load(stored(dir.path(), &uuid, "m", 2), Level::Ram).unwrap();
    let plan = Plan { steps: vec![Step { sets: vec!["m".into()], passes: 1 }] };
    // The shape measured small on an empty card.
    let r = mem.request("small", plan.clone()).unwrap();
    let pass = r.pass(m, "main", "1x8", Torch::default()).unwrap();
    pass.end(Measured { allocated_peak: REGION_ALIGN, slack: 0, sites: vec![] }, Torch::default());
    drop(r);
    card.foreign.store(8 * REGION_ALIGN, Ordering::SeqCst);
    mem.load(stored(dir.path(), &uuid, "idle", 17), Level::Gpu).unwrap();
    mem.settle();
    let r = mem.request("big", plan).unwrap();
    let pass = r.pass(m, "main", "1x8", Torch::default()).unwrap();
    let pins: Vec<varena::Pin> = (0..2).map(|b| mem.bind(m, b, 0).unwrap()).collect();
    let activation = 8 * REGION_ALIGN;
    let mut answers = vec![];
    while !card.take(activation) {
        let next = r.oom(Some(m), Some(activation), None, Torch::default());
        assert!(matches!(next, Next::Retry), "idle kept pages answer Retry: {answers:?} then {next:?} ({} free)", card.free());
        answers.push(next);
        assert!(answers.len() <= 3, "the OOM ladder ends: {answers:?}");
    }
    assert!(answers.len() >= 2, "the activation outgrew the measured room, and the driver refused the first retry: {answers:?}");
    pass.end(Measured { allocated_peak: activation, slack: 0, sites: vec![] }, Torch { allocated: activation, reserved: activation });
    drop(pins);
    card.give(activation);
    let stats = r.end();
    assert!(stats.retries >= 1 && stats.leftover == 0, "{stats:?}");
}

/// The kept-page handoff (native kept_pages_survive_a_holder_that_served, graph-core 577): pages
/// bound and released with their release events still pending are due for their Keep; a settle
/// (the planned handover before an exit) waits for those events and makes every Keep, so custody
/// holds every page when it returns.
#[test]
fn a_settle_makes_every_due_keep_before_it_returns() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("handover-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 * REGION_ALIGN };
    let _scope = Scope::new(&g.uuid);
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Arc::new(Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap());
    let s = mem.load(stored(dir.path(), &uuid, "s", 3), Level::Ram).unwrap();
    // Filled while pinned (their fills' events pending until the pins are out), then released
    // with their release events pending.
    events.pending_fences.store(true, Ordering::SeqCst);
    let pins: Vec<_> = (0..3).map(|b| mem.bind(s, b, 1).unwrap()).collect();
    events.pending_fences.store(false, Ordering::SeqCst);
    let fills = events.next.load(Ordering::SeqCst);
    (1..=fills).for_each(|e| events.complete(e));
    mem.torch_limit();
    drop(pins);
    let exports = || events.kept.lock().iter().filter(|c| **c == "export").count();
    mem.torch_limit();
    assert_eq!(exports(), 0, "kept while its consumers ran");
    let releases = fills + 1..=events.next.load(Ordering::SeqCst);
    let settling = {
        let (mem, events) = (mem.clone(), events.clone());
        std::thread::spawn(move || {
            mem.settle();
            events.log("settled");
        })
    };
    events.await_log(|log| log.contains(&"finish") || log.contains(&"settled"));
    assert!(!events.kept.lock().contains(&"settled"), "the handover waits for the consumers");
    releases.for_each(|e| events.complete(e));
    settling.join().unwrap();
    assert_eq!(exports(), 3, "every due page kept by the time settle returns");
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
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
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

