//! Reviewer's probes, step 1 (one make_room loop), over binds and ranges. Host pages.
use std::collections::HashSet;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{Dst, ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Result, SetSpec, Va};

const P: u64 = REGION_ALIGN;

fn spec(dir: &std::path::Path, uuid: &str, name: &str, pages: u64) -> SetSpec {
    spec_filled(dir, uuid, name, pages, Fill::Undeclared)
}

/// A set whose pages may be kept (`Fill::Stored`) or never (`Fill::Undeclared`).
fn spec_filled(dir: &std::path::Path, uuid: &str, name: &str, pages: u64, fill: Fill) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![7u8; (pages * P) as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: pages * P };
    let regions = (0..pages)
        .map(|i| Region { offset: i * P, nbytes: P, span: P, items: vec![Item { offset: i * P, len: P, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: i * P, len: P }), part_end: true }] })
        .collect();
    let layout = Layout {
        parts: (0..pages).map(|i| Part { what: format!("t{i}"), region: i as u32, offset: i * P, nbytes: P }).collect(),
        regions,
        nbytes: pages * P,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..pages).map(|i| vec![format!("t{i}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], fill, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Binds every block of `set` once and lets it go (filled, idle).
fn warm(mem: &Memory, set: varena::SetId, blocks: u32) {
    for b in 0..blocks {
        let pin = mem.bind(set, b, 0).unwrap();
        if pin.fresh {
            pin.filled();
        }
    }
}

/// Whether block `b` of `set` is still resident: a bind that is a hit comes back not fresh.
fn resident(mem: &Memory, set: varena::SetId, b: u32) -> bool {
    let pin = mem.bind(set, b, 0).unwrap();
    let hit = !pin.fresh;
    if pin.fresh {
        pin.filled();
    }
    hit
}

/// A card: its bytes are in use while a page lives or a torch segment holds them.
struct Card {
    host: HostBackend,
    limit: u64,
    used: AtomicU64,
    /// Release events: recorded pending while `pending` is set; complete once completed.
    pending: AtomicBool,
    next: AtomicU64,
    done: Mutex<HashSet<u64>>,
    landed: Condvar,
    /// Creates wait while held (a mechanism map behind device work).
    hold_create: Mutex<bool>,
    /// Pages exported (kept).
    exports: AtomicU64,
    create_gate: Condvar,
}

impl Card {
    fn new(limit: u64) -> Arc<Card> {
        Arc::new(Card { host: HostBackend::new(limit), limit, used: AtomicU64::new(0), pending: AtomicBool::new(false), next: AtomicU64::new(1000), done: Mutex::default(), landed: Condvar::new(), hold_create: Mutex::new(false), create_gate: Condvar::new(), exports: AtomicU64::new(0) })
    }
    fn free(&self) -> u64 {
        self.limit.saturating_sub(self.used.load(Ordering::SeqCst))
    }
    /// A torch segment, as the driver serves it: refused without room.
    fn take(&self, len: u64) -> bool {
        let used = self.used.load(Ordering::SeqCst);
        used + len <= self.limit && self.used.compare_exchange(used, used + len, Ordering::SeqCst, Ordering::SeqCst).is_ok()
    }
    fn complete(&self, e: u64) {
        self.done.lock().insert(e);
        self.landed.notify_all();
    }
}

struct Driver(Arc<Card>);

impl Backend for Driver {
    fn capacity(&self) -> u64 { self.0.limit }
    fn headroom(&self) -> u64 { self.0.host.headroom() }
    fn free(&self) -> Result<u64> { Ok(self.0.free()) }
    fn process_used(&self) -> Result<u64> { self.0.host.process_used() }
    fn opened_context(&self) -> u64 { 0 }
    fn granularity(&self) -> u64 { self.0.host.granularity() }
    fn reserve(&self, len: u64) -> Result<Va> { self.0.host.reserve(len) }
    fn release(&self, va: Va) { self.0.host.release(va) }
    fn create(&self, len: u64) -> Result<Physical> {
        {
            let mut held = self.0.hold_create.lock();
            while *held {
                self.0.create_gate.wait(&mut held);
            }
        }
        if !self.0.take(len) {
            return Err(varena::Error::Device { op: "create", code: varena::backend::OUT_OF_MEMORY, name: "out of memory".into() });
        }
        self.0.host.create(len)
    }
    fn destroy(&self, page: Physical) -> Result<()> {
        self.0.used.fetch_sub(page.len, Ordering::SeqCst);
        self.0.host.destroy(page)
    }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> { self.0.host.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> { self.0.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> { self.0.host.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> Result<Owned> {
        self.0.exports.fetch_add(1, Ordering::SeqCst);
        self.0.host.export(page)
    }
    fn import(&self, handle: Owned, len: u64) -> Result<Physical> { self.0.host.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.0.host.target(va, offset, len) }
    fn fence(&self) -> Result<Event> { self.0.host.fence() }
    fn record(&self, _: usize) -> Result<Event> {
        Ok(if self.0.pending.load(Ordering::SeqCst) { Event(self.0.next.fetch_add(1, Ordering::SeqCst)) } else { Event::DONE })
    }
    fn query(&self, e: Event) -> Result<bool> { Ok(e == Event::DONE || self.0.done.lock().contains(&e.0)) }
    fn elapsed(&self, a: Event, b: Event) -> Result<Option<u64>> { self.0.host.elapsed(a, b) }
    fn wait(&self, _: Event, _: usize) -> Result<()> { Ok(()) }
    fn synchronize(&self, stream: usize) -> Result<()> { self.0.host.synchronize(stream) }
    /// The host waits for the event, as a card's cuEventSynchronize does.
    fn finish(&self, e: Event) -> Result<()> {
        let mut done = self.0.done.lock();
        while e != Event::DONE && !done.contains(&e.0) {
            self.0.landed.wait(&mut done);
        }
        Ok(())
    }
    fn retire(&self, _: Event) {}
    fn drain(&self) -> Result<()> { Ok(()) }
}

fn open(tag: &str, pages: u64) -> (tempfile::TempDir, String, Arc<Card>, Arc<Memory>) {
    assert!(varena::os::namespace(&format!("st1{}-", std::process::id())));
    let uuid = format!("st1{tag}-{}", std::process::id());
    let card = Card::new(pages * P);
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: pages * P };
    let mem = Arc::new(Memory::open(g, Box::new(Driver(card.clone()))).unwrap());
    (tempfile::tempdir().unwrap(), uuid, card, mem)
}

fn close(uuid: &str, mem: Arc<Memory>) {
    drop(mem);
    varena::os::remove_table(uuid);
}

/// A bind's room waits on a page's release event, the state unlocked meanwhile: another
/// thread's bind proceeds, and the bind returns once the event completes, never before.
#[test]
fn make_room_waits_on_a_release_event_without_holding_the_state() {
    let (dir, uuid, card, mem) = open("wait", 2);
    let x = mem.load(spec(dir.path(), &uuid, "x", 1), Level::Ram).unwrap();
    let y = mem.load(spec(dir.path(), &uuid, "y", 1), Level::Ram).unwrap();
    let z = mem.load(spec(dir.path(), &uuid, "z", 1), Level::Ram).unwrap();
    // X let go with its release event pending (a kernel still reading it), Y idle after it.
    let pin = mem.bind(x, 0, 0).unwrap();
    if pin.fresh {
        pin.filled();
    }
    card.pending.store(true, Ordering::SeqCst);
    drop(pin);
    card.pending.store(false, Ordering::SeqCst);
    let event = card.next.load(Ordering::SeqCst) - 1;
    warm(&mem, y, 1);
    let (tx, rx) = std::sync::mpsc::channel();
    let hook = std::thread::spawn({
        let mem = mem.clone();
        move || tx.send(mem.bind(z, 0, 0).map(drop).map_err(|e| e.code())).unwrap()
    });
    std::thread::sleep(Duration::from_millis(300));
    let early = rx.try_recv().ok();
    // Another thread's state call while the hook waits.
    let (btx, brx) = std::sync::mpsc::channel();
    let other = std::thread::spawn({
        let mem = mem.clone();
        move || {
            let pin = mem.bind(y, 0, 0);
            btx.send(pin.is_ok()).unwrap();
        }
    });
    let other_done = brx.recv_timeout(Duration::from_secs(3));
    card.complete(event);
    let returned = rx.recv_timeout(Duration::from_secs(5));
    hook.join().unwrap();
    let other_after = if other_done.is_err() { brx.recv_timeout(Duration::from_secs(5)) } else { other_done };
    other.join().unwrap();
    println!("hook before the event completed: {early:?}; another bind meanwhile: {other_done:?} (then {other_after:?}); hook after it: {returned:?}");
    close(&uuid, mem);
    assert!(early.is_none(), "the hook took a page whose release event was pending");
    assert!(other_done.is_ok(), "the hook held the state while it waited on a release event");
    assert!(returned.is_ok(), "the hook never returned after the event completed");
}

/// Never a pinned page; an impossible need takes nothing; a need the idle pages cover takes
/// only them.
#[test]
fn make_room_never_takes_a_pinned_page_and_an_impossible_need_takes_nothing() {
    let (dir, uuid, _card, mem) = open("pinned", 4);
    let a = mem.load(spec(dir.path(), &uuid, "a", 2), Level::Ram).unwrap();
    let b = mem.load(spec(dir.path(), &uuid, "b", 2), Level::Ram).unwrap();
    let pins: Vec<_> = (0..2).map(|i| {
        let p = mem.bind(a, i, 0).unwrap();
        if p.fresh {
            p.filled();
        }
        p
    }).collect();
    warm(&mem, b, 2);
    let c = mem.load(spec(dir.path(), &uuid, "c", 3), Level::Ram).unwrap();
    let impossible = mem.retain(c, &[0, 1, 2]).unwrap().prepare(0).map(drop).map_err(|e| e.code());
    let b_kept = (0..2).filter(|i| resident(&mem, b, *i)).count();
    let possible = mem.retain(c, &[0, 1]).unwrap().prepare(0).map(drop).map_err(|e| e.code());
    // SAFETY: mapped pages pinned by this test; read only.
    let a_intact = pins.iter().all(|p| unsafe { *(p.ptr as *const u8) } == 7);
    drop(pins);
    println!("3 pages with 2 pinned: {impossible:?}, b's pages still resident after it: {b_kept}/2; 2 pages: {possible:?}; a's pinned bytes intact: {a_intact}");
    close(&uuid, mem);
    assert!(impossible.is_err(), "an impossible need succeeded");
    assert_eq!(b_kept, 2, "an impossible need evicted idle pages before failing");
    assert!(possible.is_ok() && a_intact, "the need the idle pages cover failed, or a pinned page was taken");
}

