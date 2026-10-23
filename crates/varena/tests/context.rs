//! The context figure (process_used less the hook's segments and varena's pages), the reviewer's probes: no drift,
//! the open publishes its sample (C1), and pages in transit are not context (C2, C3). Host pages.
#![allow(dead_code)]
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
use varena::{Access, Backend, Event, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Physical, Plan, Result, SetSpec, Step, Va};

const P: u64 = REGION_ALIGN;

fn spec(dir: &std::path::Path, uuid: &str, name: &str, pages: u64) -> SetSpec {
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
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Undeclared, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

fn plan(steps: &[&[&str]]) -> Plan {
    Plan { steps: steps.iter().map(|s| Step { sets: s.iter().map(|n| n.to_string()).collect(), passes: 1 }).collect() }
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
    /// Maps and unmaps wait while held.
    hold_map: Mutex<bool>,
    hold_unmap: Mutex<bool>,
    create_gate: Condvar,
    /// Each unmap: (VA base, offset, when).
    unmaps: Mutex<Vec<(u64, u64, std::time::Instant)>>,
    /// Each event completed: (event, when).
    completed: Mutex<Vec<(u64, std::time::Instant)>>,
}

impl Card {
    fn new(limit: u64) -> Arc<Card> {
        Arc::new(Card { host: HostBackend::new(limit), limit, used: AtomicU64::new(0), pending: AtomicBool::new(false), next: AtomicU64::new(1000), done: Mutex::default(), landed: Condvar::new(), hold_create: Mutex::new(false), hold_map: Mutex::new(false), hold_unmap: Mutex::new(false), create_gate: Condvar::new(), unmaps: Mutex::default(), completed: Mutex::default() })
    }
    fn free(&self) -> u64 {
        self.limit.saturating_sub(self.used.load(Ordering::SeqCst))
    }
    /// A torch segment, as the driver serves it: refused without room.
    fn take(&self, len: u64) -> bool {
        let used = self.used.load(Ordering::SeqCst);
        used + len <= self.limit && self.used.compare_exchange(used, used + len, Ordering::SeqCst, Ordering::SeqCst).is_ok()
    }
    fn give(&self, len: u64) {
        self.used.fetch_sub(len, Ordering::SeqCst);
    }
    fn complete(&self, e: u64) {
        self.completed.lock().push((e, std::time::Instant::now()));
        self.done.lock().insert(e);
        self.landed.notify_all();
    }
}

struct Driver(Arc<Card>);

impl Backend for Driver {
    fn capacity(&self) -> u64 { self.0.limit }
    fn free(&self) -> Result<u64> { Ok(self.0.free()) }
    fn process_used(&self) -> Result<u64> { Ok(self.0.used.load(Ordering::SeqCst)) }
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
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> Result<()> {
        {
            let mut held = self.0.hold_map.lock();
            while *held {
                self.0.create_gate.wait(&mut held);
            }
        }
        self.0.host.map(va, offset, page, access)
    }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> Result<()> { self.0.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> Result<()> {
        {
            let mut held = self.0.hold_unmap.lock();
            while *held {
                self.0.create_gate.wait(&mut held);
            }
        }
        self.0.unmaps.lock().push((va.base, offset, std::time::Instant::now()));
        self.0.host.unmap(va, offset, len)
    }
    fn export(&self, page: &Physical) -> Result<Owned> { self.0.host.export(page) }
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
    let mem = Arc::new(Memory::open(g, None, Box::new(Driver(card.clone()))).unwrap());
    (tempfile::tempdir().unwrap(), uuid, card, mem)
}

fn close(uuid: &str, mem: Arc<Memory>) {
    drop(mem);
    varena::os::remove_table(uuid);
}



fn bytes(b: u64) -> Vec<u8> {
    (0..P).map(|i| (i as u8).wrapping_mul(31).wrapping_add(b as u8)).collect()
}

/// A stored set of `blocks` pages, each its own bytes.
fn pattern(dir: &std::path::Path, uuid: &str, name: &str, blocks: u64) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, (0..blocks).flat_map(bytes).collect::<Vec<u8>>()).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: blocks * P };
    let layout = Layout {
        parts: (0..blocks).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * P, nbytes: P }).collect(),
        regions: (0..blocks).map(|r| Region { offset: r * P, nbytes: P, span: P, items: vec![Item { offset: r * P, len: P, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * P, len: P }), part_end: true }] }).collect(),
        nbytes: blocks * P,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..blocks).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

/// Binds every block of `set` once, in order, checking its bytes; returns each block's address.
fn sweep(mem: &Memory, set: varena::SetId, blocks: u32) -> std::result::Result<Vec<u64>, String> {
    (0..blocks).map(|b| {
        let pin = mem.bind(set, b, 0).map_err(|e| format!("block {b}: {e}"))?;
        // SAFETY: a host page mapped at `ptr`, pinned for this read.
        let got = unsafe { std::slice::from_raw_parts(pin.ptr as *const u8, pin.len as usize) };
        if got != bytes(u64::from(b)).as_slice() {
            return Err(format!("block {b} has other bytes"));
        }
        Ok(pin.ptr)
    }).collect()
}
/// A request's open samples the device (after its own publish): its end publishes that sample.
fn sampled(mem: &Memory, ledger: &varena::Ledger, name: &str) -> u64 {
    drop(mem.request(name, Plan::default()).unwrap());
    context(ledger)
}

/// This row's context as the ledger publishes it.
fn context(ledger: &varena::Ledger) -> u64 {
    let me = std::process::id();
    ledger.status().rows.iter().filter(|r| r.pid == me && r.live).map(|r| r.context).sum()
}

fn opened(tag: &str, pages: u64) -> (tempfile::TempDir, String, Arc<Card>, varena::Ledger, Arc<Memory>) {
    assert!(varena::os::namespace(&format!("st1{}-", std::process::id())));
    let uuid = format!("ctx{tag}-{}", std::process::id());
    let card = Card::new(pages * P);
    let g = GpuId { uuid: uuid.clone(), model: "host".into(), memory: pages * P };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Arc::new(Memory::open(g, None, Box::new(Driver(card.clone()))).unwrap());
    (tempfile::tempdir().unwrap(), uuid, card, ledger, mem)
}

/// The context is exactly what the process holds outside varena's pages and the hook's segments:
/// a raw allocation (as the native suite's Pressure::fill) of 3 pages reads 3 pages, and churn
/// (streaming passes, hook segments made and freed) leaves it there, measured at each request's
/// open: no drift that would hide an accounting error.
#[test]
fn the_context_is_the_bytes_outside_pages_and_segments_and_never_drifts() {
    let (dir, uuid, card, ledger, mem) = opened("drift", 12);
    let w = mem.load(pattern(dir.path(), &uuid, "w", 10), Level::Ram).unwrap();
    assert!(card.take(3 * P), "the raw allocation");
    let first = mem.request("first", plan(&[&["w"]])).unwrap();
    let at_open = context(&ledger);
    drop(first);
    println!("published while the first request is open: {} pages; at its end: {} pages", at_open as f64 / P as f64, context(&ledger) as f64 / P as f64);
    assert_eq!(at_open, 3 * P, "the open published before its sample (C1)");
    let mut seen = vec![];
    for round in 0..4 {
        // Its open samples the device.
        seen.push(sampled(&mem, &ledger, &format!("s{round}")));
        let r = mem.request(&format!("r{round}"), plan(&[&["w"]])).unwrap();
        // A pass over the set (it streams on 12 pages less the raw 3), and a hook segment made and freed.
        sweep(&mem, w, 10).unwrap();
        if let Ok(claim) = mem.make_room(P) {
            if card.take(P) {
                let segment = claim.resident();
                drop(segment);
                card.give(P);
            }
        }
        drop(r);
    }
    seen.push(sampled(&mem, &ledger, "last"));
    println!("context at each request's open (pages): {:?}", seen.iter().map(|c| *c as f64 / P as f64).collect::<Vec<_>>());
    close(&uuid, mem);
    assert!(seen.iter().all(|c| *c == 3 * P), "the context drifted from the 3 raw pages: {seen:?}");
}

/// A slot moving (its page between its old block and its new one, in a Move) is in neither a
/// home nor a retirement: a sample meanwhile counts it as context.
#[test]
fn a_slot_in_transit_is_not_context() {
    let (dir, uuid, card, ledger, mem) = opened("transit", 6);
    let w = mem.load(pattern(dir.path(), &uuid, "w", 10), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["w"]])).unwrap();
    sweep(&mem, w, 10).unwrap();
    // The next pass's first upper block moves a slot: its map is held.
    *card.hold_map.lock() = true;
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || sweep(&mem, w, 10).map(|_| ())
    });
    std::thread::sleep(Duration::from_millis(300));
    // A sample meanwhile (another request's open).
    assert!(!binder.is_finished(), "the pass did not wait on a held map");
    let during = sampled(&mem, &ledger, "other");
    assert!(!binder.is_finished(), "the pass did not wait on a held map");
    *card.hold_map.lock() = false;
    card.create_gate.notify_all();
    let pass = binder.join().unwrap();
    let settled = sampled(&mem, &ledger, "after");
    drop(r);
    println!("context while a slot moves: {} pages; after: {} pages; the pass {pass:?}", during as f64 / P as f64, settled as f64 / P as f64);
    close(&uuid, mem);
    assert_eq!(during, 0, "a slot in transit was counted as context");
}

/// A page the hook takes in place (its backing out of its home while it unmaps on the hook's
/// thread) is in neither a home nor a retirement: a sample meanwhile counts it as context.
#[test]
fn a_page_the_hook_is_taking_is_not_context() {
    let (dir, uuid, card, ledger, mem) = opened("taking", 4);
    let x = mem.load(pattern(dir.path(), &uuid, "x", 4), Level::Ram).unwrap();
    // Open: its pages are not kept (the hook takes no kept page).
    let r = mem.request("r", plan(&[&["x"]])).unwrap();
    sweep(&mem, x, 4).unwrap();
    *card.hold_unmap.lock() = true;
    let hook = std::thread::spawn({
        let mem = mem.clone();
        move || mem.make_room(P).map(drop).map_err(|e| e.code())
    });
    std::thread::sleep(Duration::from_millis(300));
    assert!(!hook.is_finished(), "the hook did not wait on a held unmap");
    let during = sampled(&mem, &ledger, "other");
    assert!(!hook.is_finished(), "the hook did not wait on a held unmap");
    *card.hold_unmap.lock() = false;
    card.create_gate.notify_all();
    let made = hook.join().unwrap();
    drop(r);
    println!("context while the hook takes a page in place: {} pages; make_room {made:?}", during as f64 / P as f64);
    close(&uuid, mem);
    assert_eq!(during, 0, "a page the hook is taking was counted as context");
}
