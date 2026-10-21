//! Ordering against device events: a page whose transform is still running on the bind stream
//! is never made read-only or exported, and a pin's release is not complete before its event.
//! Host pages fill synchronously, so the backend here is the host backend with stream events
//! that complete only when the test says (as a card's do when their work finishes).

use std::collections::HashSet;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
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
        self.host.free()
    }
    fn process_used(&self) -> Result<u64> {
        self.host.process_used()
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
        Ok(Event::DONE)
    }
    /// A stream event: pending until the test completes it.
    fn record(&self, _: usize) -> Result<Event> {
        Ok(Event(self.events.next.fetch_add(1, Ordering::Relaxed) + 1))
    }
    fn query(&self, event: Event) -> Result<bool> {
        Ok(event == Event::DONE || self.events.done.lock().contains(&event.0))
    }
    fn elapsed(&self, _: Event, _: Event) -> Result<Option<u64>> {
        Ok(Some(0))
    }
    fn wait(&self, _: Event, _: usize) -> Result<()> {
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
    let pass = r.pass(id, "1x1", Torch::default()).unwrap();
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
