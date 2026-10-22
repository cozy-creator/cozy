//! Wakes without the state lock (S1 review 5, F3): a pin's drop that lands between a binder's
//! last check and its sleep still wakes it. The reviewer's probe, as a test.
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Result, SetSpec, Va};

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

