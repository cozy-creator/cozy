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
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Plan, Result, SetSpec, Step, Torch, Va};

#[derive(Default)]
struct Events {
    next: AtomicU64,
    done: Mutex<HashSet<u64>>,
    /// Calls that make a page read-only or hand it out.
    kept: Mutex<Vec<&'static str>>,
    unmaps: AtomicU64,
    /// In-stream waits asked for, in order: (event, stream).
    waits: Mutex<Vec<(u64, usize)>>,
    /// Fences are pending events too (a card's copy), and a fence may hold its fill thread
    /// until a gate opens (a long copy).
    pending_fences: AtomicBool,
    fences: Mutex<Vec<u64>>,
    hold_fence: AtomicBool,
    fence_gate: Gate,
    /// The next event record fails (a driver error), and the streams synchronized instead.
    fail_record: AtomicBool,
    syncs: Mutex<Vec<usize>>,
    /// Another process's bytes on the device, and bytes that land there at the next create
    /// (between a decision and its page).
    foreign: AtomicU64,
    lands: AtomicU64,
    /// The process's own bytes cannot be measured (WDDM); open measured this context.
    unmeasured: AtomicU64,
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
        let landed = self.events.lands.swap(0, Ordering::SeqCst);
        let foreign = self.events.foreign.fetch_add(landed, Ordering::SeqCst) + landed;
        if self.host.free()? < len + foreign {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        self.host.create(len)
    }
    fn destroy(&self, page: Physical) {
        self.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        self.host.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        self.events.kept.lock().push("protect");
        self.host.protect(va, offset, len, access)
    }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        self.events.unmaps.fetch_add(1, Ordering::Relaxed);
        self.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> {
        self.events.kept.lock().push("export");
        self.host.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        self.host.import(handle, len)
    }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
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
        Ok(event == Event::DONE || self.events.done.lock().contains(&event.0))
    }
    fn elapsed(&self, _: Event, _: Event) -> Result<Option<u64>> {
        Ok(Some(0))
    }
    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        if event != Event::DONE {
            self.events.waits.lock().push((event.0, stream));
        }
        Ok(())
    }
    fn synchronize(&self, stream: usize) -> Result<()> {
        self.events.syncs.lock().push(stream);
        Ok(())
    }
    fn register_host(&self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn unregister_host(&self, _: u64) -> Result<()> {
        Ok(())
    }
    fn retire(&self, _: Event) {}
    fn drain(&self) -> Result<()> {
        Ok(())
    }
    unsafe fn register_host(&self, ptr: *const u8, len: u64) -> Result<()> {
        // SAFETY: the caller's contract, passed on.
        unsafe { self.host.register_host(ptr, len) }
    }
    fn unregister_host(&self, ptr: *const u8) -> Result<()> {
        self.host.unregister_host(ptr)
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
    events.done.lock().insert(transform);
    mem.torch_limit();
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
    let _ledger = varena::Ledger::create(&g, None).unwrap();
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
    events.done.lock().insert(release);
    mem.torch_limit();
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
            events.done.lock().insert(release);
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
    let addr = SocketAddr::from_abstract_name(format!("cozy-varena-{uid}-custody-{gpu}")).unwrap();
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

/// S2 review, 6a: a release whose event cannot be recorded never reads as complete: that one
/// pin's stream is synchronized.
#[test]
fn a_release_without_an_event_synchronizes_its_stream() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("record-{}", std::process::id()), model: "host".into(), memory: 8 << 20 };
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events: events.clone() })).unwrap();
    let id = mem.load(set(dir.path(), &g.uuid), Level::Ram).unwrap();
    let pin = mem.bind(id, 0, 7).unwrap();
    events.fail_record.store(true, Ordering::SeqCst);
    drop(pin);
    assert_eq!(*events.syncs.lock(), vec![7]);
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
/// process's own or custody's (here the fake custody's, in this process).
struct Driver {
    host: HostBackend,
    limit: u64,
    /// id → (inode, bytes, destroyed by this process).
    pages: Mutex<std::collections::HashMap<u64, (u64, u64, bool)>>,
}

fn inode(h: &Owned) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::File::from(h.try_clone().unwrap()).metadata().unwrap().ino()
}

impl Driver {
    fn used(&self) -> u64 {
        let open: HashSet<u64> = std::fs::read_dir("/proc/self/fd").unwrap().flatten().filter_map(|e| std::fs::metadata(e.path()).ok()).map(|m| std::os::unix::fs::MetadataExt::ino(&m)).collect();
        self.pages.lock().values().filter(|(ino, _, gone)| !gone || open.contains(ino)).map(|(_, len, _)| len).sum()
    }
}

impl Backend for Driver {
    fn capacity(&self) -> u64 {
        self.limit
    }
    fn free(&self) -> Result<u64> {
        Ok(self.limit.saturating_sub(self.used()))
    }
    fn process_used(&self) -> Result<u64> {
        self.host.process_used()
    }
    fn opened_context(&self) -> u64 {
        0
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
        if self.used() + len > self.limit {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        let page = self.host.create(len)?;
        let ino = inode(&self.host.export(&page)?);
        self.pages.lock().insert(page.id, (ino, len, false));
        Ok(page)
    }
    fn destroy(&self, page: Physical) {
        if let Some(p) = self.pages.lock().get_mut(&page.id) {
            p.2 = true;
        }
        self.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        self.host.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> {
        self.host.protect(va, offset, len, access)
    }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        self.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> {
        self.host.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> {
        self.host.import(handle, len)
    }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst {
        self.host.target(va, offset, len)
    }
    fn fence(&self) -> Result<Event> {
        self.host.fence()
    }
    fn record(&self, stream: usize) -> Result<Event> {
        self.host.record(stream)
    }
    fn query(&self, event: Event) -> Result<bool> {
        self.host.query(event)
    }
    fn elapsed(&self, from: Event, to: Event) -> Result<Option<u64>> {
        self.host.elapsed(from, to)
    }
    fn wait(&self, event: Event, stream: usize) -> Result<()> {
        self.host.wait(event, stream)
    }
    fn synchronize(&self, stream: usize) -> Result<()> {
        self.host.synchronize(stream)
    }
    fn register_host(&self, ptr: u64, len: u64) -> Result<()> {
        self.host.register_host(ptr, len)
    }
    fn unregister_host(&self, ptr: u64) -> Result<()> {
        self.host.unregister_host(ptr)
    }
    fn retire(&self, event: Event) {
        self.host.retire(event)
    }
    fn drain(&self) -> Result<()> {
        self.host.drain()
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
    let (gate, log) = (Arc::new(Gate::default()), Arc::new(Log::default()));
    gate.open();
    serve_custody(&uuid, gate, log.clone());
    let driver = Driver { host: HostBackend::new(g.memory), limit: g.memory, pages: Mutex::new(Default::default()) };
    let mem = Memory::open(g.clone(), None, Box::new(driver)).unwrap();
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
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let events = Arc::new(Events::default());
    events.unmeasured.store(5 << 20, Ordering::SeqCst);
    let mem = Memory::open(g.clone(), None, Box::new(Stream { host: HostBackend::new(g.memory), events })).unwrap();
    mem.torch_limit();
    let pid = std::process::id();
    assert_eq!(ledger.status().rows.iter().find(|r| r.pid == pid && r.context > 0).map(|r| r.context), Some(5 << 20));
    drop(mem);
}

/// Review 3, N6: a kept page whose fill identity changed is not overwritten in place while
/// custody still holds it under the old key: the fill waits until custody has let go, so a
/// successor (after a kill -9 at any point) never imports new bytes under the old key.
#[test]
fn a_kept_page_is_never_overwritten_under_its_old_key() {
    let dir = tempfile::tempdir().unwrap();
    let uuid = format!("rekey-{}", std::process::id());
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: 8 << 20 };
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
    let (ptr, len) = (pin.ptr, pin.len as usize);
    pin.filled();
    events.done.lock().insert(events.next.load(Ordering::Relaxed));
    mem.torch_limit();
    log.await_call(|(_, c)| c.contains("Export") && c.contains(&old));
    // Its consumer's stream still reads it; custody stops answering; lora-b is loaded.
    drop(pin);
    gate.close();
    let mut b = spec;
    b.fingerprint.fill = Fill::Declared("lora-b".into());
    mem.load(b, Level::Ram).unwrap();
    let binder = {
        let mem = mem.clone();
        std::thread::spawn(move || mem.bind(id, 0, 2).unwrap().fresh)
    };
    log.await_call(|(_, c)| c.contains("Withdraw") && c.contains(&old));
    // SAFETY: the page stays mapped in place; read only.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    assert!(bytes.iter().all(|b| *b == 0xab), "overwritten while custody kept it under the old key");
    gate.open();
    assert!(binder.join().unwrap(), "filled again, fresh for lora-b");
    assert!(!log.kept.lock().keys().any(|(fp, _)| *fp == old));
}
