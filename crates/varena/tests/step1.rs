//! Reviewer's probes, step 1 (one make_room loop). Host pages; run under `unshare --user --net`.
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
    create_gate: Condvar,
}

impl Card {
    fn new(limit: u64) -> Arc<Card> {
        Arc::new(Card { host: HostBackend::new(limit), limit, used: AtomicU64::new(0), pending: AtomicBool::new(false), next: AtomicU64::new(1000), done: Mutex::default(), landed: Condvar::new(), hold_create: Mutex::new(false), create_gate: Condvar::new() })
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

/// The hook waits on a page's release event, the state unlocked meanwhile: another thread's
/// bind proceeds, and the hook returns once the event completes, never before.
#[test]
fn make_room_waits_on_a_release_event_without_holding_the_state() {
    let (dir, uuid, card, mem) = open("wait", 2);
    let x = mem.load(spec(dir.path(), &uuid, "x", 1), Level::Ram).unwrap();
    let y = mem.load(spec(dir.path(), &uuid, "y", 1), Level::Ram).unwrap();
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
        move || tx.send(mem.make_room(P).map(drop).map_err(|e| e.code())).unwrap()
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
    let impossible = mem.make_room(3 * P).map(drop).map_err(|e| e.code());
    let b_kept = (0..2).filter(|i| resident(&mem, b, *i)).count();
    let possible = mem.make_room(2 * P).map(drop).map_err(|e| e.code());
    // SAFETY: mapped pages pinned by this test; read only.
    let a_intact = pins.iter().all(|p| unsafe { *(p.ptr as *const u8) } == 7);
    drop(pins);
    println!("3 pages with 2 pinned: {impossible:?}, b's pages still resident after it: {b_kept}/2; 2 pages: {possible:?}; a's pinned bytes intact: {a_intact}");
    close(&uuid, mem);
    assert!(impossible.is_err(), "an impossible need succeeded");
    assert_eq!(b_kept, 2, "an impossible need evicted idle pages before failing");
    assert!(possible.is_ok() && a_intact, "the need the idle pages cover failed, or a pinned page was taken");
}

/// The order: another set's idle page (N, least recently used), then a later step's (D), then
/// caches nobody holds, then the current step's (W), furthest next use first.
#[test]
fn make_room_takes_idle_then_caches_then_furthest_declared() {
    let (dir, uuid, card, mem) = open("order", 5);
    let a = mem.load(spec(dir.path(), &uuid, "a", 1), Level::Ram).unwrap();
    let b = mem.load(spec(dir.path(), &uuid, "b", 1), Level::Ram).unwrap();
    let n = mem.load(spec(dir.path(), &uuid, "n", 1), Level::Ram).unwrap();
    let shrunk = Arc::new(AtomicU64::new(0));
    let cache = mem.cache("graphs", {
        let shrunk = shrunk.clone();
        move |want| {
            shrunk.fetch_add(want, Ordering::SeqCst);
            want
        }
    });
    warm(&mem, n, 1);
    warm(&mem, a, 1);
    warm(&mem, b, 1);
    let r = mem.request("r", plan(&[&["a"], &["b"]])).unwrap();
    // The cache holds one page of torch's bytes (counted as a segment of this process).
    assert!(card.take(P));
    let _segment = mem.resident(P);
    mem.cached(cache, P);
    let free = card.free();
    let first = mem.make_room(free + P).map(drop).map_err(|e| e.code());
    let after_first = (resident(&mem, n, 0), shrunk.load(Ordering::SeqCst), resident(&mem, b, 0), resident(&mem, a, 0));
    println!("free {free}; one page more: {first:?}; then (n resident, cache shrunk, b resident, a resident) = {after_first:?}");
    drop(r);
    close(&uuid, mem);
    assert!(!after_first.0 && after_first.1 == 0 && after_first.2 && after_first.3, "the first page taken was not the idle N page");
}

/// The lead's ruling (S4's v14 graphs): a streaming set's room comes from another set's idle
/// weight pages, though a later step uses them, before that set's graph cache. A page refills from
/// the page cache; a graph recaptures far slower.
#[test]
fn a_streaming_sets_room_takes_another_sets_idle_pages_before_its_graph_cache() {
    let (dir, uuid, card, mem) = open("graphorder", 5);
    let dit = mem.load(spec(dir.path(), &uuid, "dit", 3), Level::Ram).unwrap();
    let vae = mem.load(spec(dir.path(), &uuid, "vae", 2), Level::Ram).unwrap();
    warm(&mem, vae, 2);
    let shrunk = Arc::new(AtomicU64::new(0));
    let cache = mem.cache("vae-graphs", {
        let shrunk = shrunk.clone();
        move |want| {
            shrunk.fetch_add(want, Ordering::SeqCst);
            want
        }
    });
    // The VAE's captured graphs: one page of torch's segments, touched after its weights.
    assert!(card.take(P));
    let _segment = mem.resident(P);
    mem.cached(cache, P);
    let r = mem.request("r", plan(&[&["dit"], &["vae"]])).unwrap();
    let pass = r.pass(dit, "k").unwrap();
    let pins: Vec<varena::Pin> = (0..3).map(|b| mem.bind(dit, b, 0).unwrap()).collect();
    let shrunk = shrunk.load(Ordering::SeqCst);
    drop(pins);
    pass.end(0);
    r.end();
    let vae_left = (0..2).filter(|b| resident(&mem, vae, *b)).count();
    close(&uuid, mem);
    assert_eq!((shrunk, vae_left), (0, 1), "(cache bytes shrunk, VAE pages left): one VAE page goes, the graphs stay");
}

/// The bytes make_room freed are nobody's: another thread's bind maps into them before the
/// hook's segment, which the driver then refuses, though an idle page was left to take.
#[test]
fn the_room_the_hook_made_is_not_held_for_its_segment() {
    let (dir, uuid, card, mem) = open("steal", 2);
    let x = mem.load(spec(dir.path(), &uuid, "x", 1), Level::Ram).unwrap();
    let y = mem.load(spec(dir.path(), &uuid, "y", 1), Level::Ram).unwrap();
    let z = mem.load(spec(dir.path(), &uuid, "z", 1), Level::Ram).unwrap();
    warm(&mem, x, 1);
    warm(&mem, y, 1);
    // The hook: room for one page of torch's, claimed for its segment.
    let made = mem.make_room(P);
    // Meanwhile (between the hook's make_room and its segment) another thread binds a page.
    let bound = mem.bind(z, 0, 0).map(|p| {
        if p.fresh {
            p.filled();
        }
    });
    let segment = card.take(P);
    let _resident = match (&made, segment) {
        (Ok(_), true) => made.ok().map(|c| c.resident()),
        _ => None,
    };
    let made = _resident.as_ref().map(|_| ());
    let evictable_left = resident(&mem, y, 0) || resident(&mem, z, 0);
    println!("claim held: {made:?}; another bind {:?}; the hook's segment served: {segment}; an idle page was left to take: {evictable_left}", bound.is_ok());
    close(&uuid, mem);
    assert!(segment || !evictable_left, "the hook's segment was refused while an idle page was left");
}

/// The hook runs caches' shrinks (the caller's code: Python, the GIL, torch's frees) inside the
/// allocation, on the allocating thread.
#[test]
fn make_room_runs_a_caches_shrink_inside_the_hook() {
    let (_dir, uuid, card, mem) = open("shrink", 2);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let cache = mem.cache("graphs", {
        let calls = calls.clone();
        move |want| {
            calls.lock().push(std::thread::current().name().unwrap_or("?").to_string());
            want
        }
    });
    assert!(card.take(2 * P));
    let _segments = mem.resident(2 * P);
    mem.cached(cache, 2 * P);
    let hook = std::thread::Builder::new().name("torch-allocating-thread".into()).spawn({
        let mem = mem.clone();
        move || mem.make_room(P).map(drop).map_err(|e| e.code())
    }).unwrap();
    let made = hook.join().unwrap();
    let calls = calls.lock().clone();
    println!("make_room {made:?}; shrink callbacks run on: {calls:?}");
    close(&uuid, mem);
    assert!(calls.is_empty(), "the allocator hook ran a cache's shrink (caller code) inside the allocation");
}

/// Ruling 2's other half: caches still shrink at a bind (outside the hook), on the binding
/// thread, so a bind short of room with only a cache to give is served.
#[test]
fn a_bind_still_shrinks_a_cache_for_its_room() {
    let (dir, uuid, card, mem) = open("bindshrink", 2);
    let x = mem.load(spec(dir.path(), &uuid, "x", 1), Level::Ram).unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    assert!(card.take(2 * P));
    // Torch's segments holding the cache: the shrink frees them (the driver and the token).
    let segments = Arc::new(Mutex::new(Some(mem.resident(2 * P))));
    let card2 = card.clone();
    let cache = mem.cache("graphs", {
        let (calls, segments) = (calls.clone(), segments.clone());
        move |want| {
            calls.lock().push(std::thread::current().name().unwrap_or("?").to_string());
            if segments.lock().take().is_some() {
                card2.used.fetch_sub(2 * P, Ordering::SeqCst);
            }
            want
        }
    });
    mem.cached(cache, 2 * P);
    let bound = std::thread::Builder::new().name("binding-thread".into()).spawn({
        let mem = mem.clone();
        move || mem.bind(x, 0, 0).map(|p| {
            if p.fresh {
                p.filled();
            }
        }).map_err(|e| e.code())
    }).unwrap().join().unwrap();
    let calls = calls.lock().clone();
    println!("bind with only a cache to give: {bound:?}; shrinks on {calls:?}");
    segments.lock().take();
    close(&uuid, mem);
    assert!(bound.is_ok() && calls.iter().all(|t| t == "binding-thread") && !calls.is_empty(), "a bind no longer shrinks caches for its room");
}

/// The hook's unmap waits for the mechanism thread's queue up to it (`settle_to`): a Map queued
/// before it for another thread's bind, stuck behind device work (as cuMemSetAccess can be),
/// holds the allocation though the hook's own page is idle.
#[test]
fn the_hooks_unmap_waits_behind_an_unrelated_map() {
    let (dir, uuid, card, mem) = open("mech", 2);
    let x = mem.load(spec(dir.path(), &uuid, "x", 1), Level::Ram).unwrap();
    let z = mem.load(spec(dir.path(), &uuid, "z", 1), Level::Ram).unwrap();
    warm(&mem, x, 1);
    *card.hold_create.lock() = true;
    // Another thread binds z: its Map is queued and stuck in create.
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || drop(mem.bind(z, 0, 0).map(|p| {
            if p.fresh {
                p.filled();
            }
        }))
    });
    std::thread::sleep(Duration::from_millis(200));
    let (tx, rx) = std::sync::mpsc::channel();
    let hook = std::thread::spawn({
        let mem = mem.clone();
        move || tx.send(mem.make_room(P).map(drop).map_err(|e| e.code())).unwrap()
    });
    let within = rx.recv_timeout(Duration::from_secs(2));
    *card.hold_create.lock() = false;
    card.create_gate.notify_all();
    let after = if within.is_err() { rx.recv_timeout(Duration::from_secs(5)).ok() } else { None };
    hook.join().unwrap();
    binder.join().unwrap();
    println!("the hook's make_room with an unrelated Map stuck ahead: within 2 s {within:?}; after the map ran {after:?}");
    close(&uuid, mem);
    assert!(within.is_ok(), "the hook waited behind an unrelated mechanism map");
}

/// F1 on the watermark's path: the hook streams the running set's upper pages in place, on its own thread, so it never
/// waits behind an unrelated mechanism map either. A request runs over a set that fills the card. Another thread's
/// bind streams one of its pages out through the mechanism, then sticks in create. The hook's make_room is still
/// served at once, from the set's next upper page.
#[test]
fn the_hooks_streaming_never_waits_behind_an_unrelated_map() {
    let (dir, uuid, card, mem) = open("streammech", 6);
    let x = mem.load(spec(dir.path(), &uuid, "x", 6), Level::Ram).unwrap();
    let z = mem.load(spec(dir.path(), &uuid, "z", 1), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["x"]])).unwrap();
    let pass = r.pass(x, "fwd|1").unwrap();
    warm(&mem, x, 6);
    *card.hold_create.lock() = true;
    // Another thread binds z: its room comes from x streaming, then its Map is queued and stuck in create.
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || drop(mem.bind(z, 0, 0).map(|p| {
            if p.fresh {
                p.filled();
            }
        }))
    });
    std::thread::sleep(Duration::from_millis(200));
    let (tx, rx) = std::sync::mpsc::channel();
    let hook = std::thread::spawn({
        let mem = mem.clone();
        move || tx.send(mem.make_room(P).map(drop).map_err(|e| e.code())).unwrap()
    });
    let within = rx.recv_timeout(Duration::from_secs(2));
    *card.hold_create.lock() = false;
    card.create_gate.notify_all();
    let after = if within.is_err() { rx.recv_timeout(Duration::from_secs(5)).ok() } else { None };
    hook.join().unwrap();
    binder.join().unwrap();
    println!("the hook's streaming make_room with an unrelated Map stuck ahead: within 2 s {within:?}; after the map ran {after:?}");
    pass.end(0);
    drop(r);
    close(&uuid, mem);
    assert!(matches!(within, Ok(Ok(()))), "the hook's streaming waited behind an unrelated mechanism map, or took nothing");
}

/// F3's rule: the hook never waits on, or takes, a page being filled. With nothing else to take
/// and another thread's page stuck filling (its map behind device work, its read on disk), the
/// hook returns at once, refused.
#[test]
fn the_hook_never_waits_on_a_filling_page() {
    let (dir, uuid, card, mem) = open("filling", 1);
    let z = mem.load(spec(dir.path(), &uuid, "z", 1), Level::Ram).unwrap();
    *card.hold_create.lock() = true;
    let binder = std::thread::spawn({
        let mem = mem.clone();
        move || drop(mem.bind(z, 0, 0).map(|p| {
            if p.fresh {
                p.filled();
            }
        }))
    });
    std::thread::sleep(Duration::from_millis(200));
    let (tx, rx) = std::sync::mpsc::channel();
    let hook = std::thread::spawn({
        let mem = mem.clone();
        move || tx.send(mem.make_room(P).map(drop).map_err(|e| e.code())).unwrap()
    });
    let within = rx.recv_timeout(Duration::from_secs(2));
    *card.hold_create.lock() = false;
    card.create_gate.notify_all();
    let after = if within.is_err() { rx.recv_timeout(Duration::from_secs(5)).ok() } else { None };
    hook.join().unwrap();
    binder.join().unwrap();
    println!("the hook with only a filling page around: within 2 s {within:?}; after the fill {after:?}");
    close(&uuid, mem);
    assert!(within.is_ok(), "the hook waited on another thread's fill");
}
