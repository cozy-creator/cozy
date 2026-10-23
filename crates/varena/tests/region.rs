//! §11 regions through Memory, as S3's bindings call them: a cache region is reclaimed only while
//! unpinned, and its next pin maps it again at the same addresses (fresh when discarded, with its
//! bytes when swapped); a graph region's drop waits for every stream's last released replay (R2-3),
//! even racing one; room's tiers. Events are the test's own: one completes when the test says.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::io::Dst;
use varena::arena::{PoolId, CHUNK};
use varena::region::Reclaimed;
use varena::{Access, Backend, Event, HostBackend, Memory, Physical, RegionKind, Va};

const MB: u64 = 1 << 20;
const A: usize = 0xa00;
const B: usize = 0xb00;
/// A stream only graph replays use.
const G: usize = 0x900;

/// Host pages whose events complete when the test says (or at once, `instant`); `finish` waits.
struct Dev {
    host: HostBackend,
    instant: AtomicBool,
    next: AtomicU64,
    events: Mutex<HashMap<u64, (usize, bool)>>,
    landed: Condvar,
    /// Every event recorded, in order: (stream, id).
    recorded: Mutex<Vec<(usize, u64)>>,
    /// Every event a finish waited for, once complete.
    finished: Mutex<Vec<u64>>,
    /// The stream a finish waits on now.
    waiting: Mutex<Option<usize>>,
}

impl Dev {
    fn complete(&self, stream: usize) {
        self.events.lock().values_mut().filter(|(s, _)| *s == stream).for_each(|(_, d)| *d = true);
        self.landed.notify_all();
    }

    fn last(&self, stream: usize) -> Option<u64> {
        self.recorded.lock().iter().rev().find(|(s, _)| *s == stream).map(|(_, e)| *e)
    }
}

/// The test keeps a handle on the device Memory owns.
struct Shared(Arc<Dev>);

impl Backend for Shared {
    fn capacity(&self) -> u64 { self.0.host.capacity() }
    fn free(&self) -> varena::Result<u64> { self.0.host.free() }
    fn process_used(&self) -> varena::Result<u64> { self.0.host.process_used() }
    fn opened_context(&self) -> u64 { 0 }
    fn granularity(&self) -> u64 { self.0.host.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.0.host.reserve(len) }
    fn release(&self, va: Va) { self.0.host.release(va) }
    fn create(&self, len: u64) -> varena::Result<Physical> { self.0.host.create(len) }
    fn destroy(&self, page: Physical) -> varena::Result<()> { self.0.host.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> { self.0.host.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> { self.0.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> { self.0.host.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> varena::Result<Owned> { self.0.host.export(page) }
    fn import(&self, handle: Owned, len: u64) -> varena::Result<Physical> { self.0.host.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.0.host.target(va, offset, len) }
    fn copy_out(&self, va: Va, offset: u64, dst: &mut [u8], after: Event) -> varena::Result<()> { self.0.host.copy_out(va, offset, dst, after) }
    fn copy_in(&self, va: Va, offset: u64, src: &[u8]) -> varena::Result<()> { self.0.host.copy_in(va, offset, src) }
    fn fence(&self) -> varena::Result<Event> { Ok(Event::DONE) }
    fn record(&self, stream: usize) -> varena::Result<Event> {
        let id = self.0.next.fetch_add(1, SeqCst);
        self.0.events.lock().insert(id, (stream, self.0.instant.load(SeqCst)));
        self.0.recorded.lock().push((stream, id));
        Ok(Event(id))
    }
    fn query(&self, event: Event) -> varena::Result<bool> {
        Ok(event == Event::DONE || self.0.events.lock().get(&event.0).is_none_or(|e| e.1))
    }
    fn elapsed(&self, _: Event, _: Event) -> varena::Result<Option<u64>> { Ok(Some(0)) }
    fn wait(&self, _: Event, _: usize) -> varena::Result<()> { Ok(()) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> {
        self.0.complete(stream);
        Ok(())
    }
    fn finish(&self, event: Event) -> varena::Result<()> {
        let mut events = self.0.events.lock();
        while let Some(&(stream, false)) = events.get(&event.0) {
            *self.0.waiting.lock() = Some(stream);
            self.0.landed.wait(&mut events);
        }
        *self.0.waiting.lock() = None;
        self.0.finished.lock().push(event.0);
        Ok(())
    }
    fn retire(&self, _: Event) {}
    fn drain(&self) -> varena::Result<()> { Ok(()) }
}

static OPENS: AtomicUsize = AtomicUsize::new(0);

/// A standalone Memory on its own host device, torch served.
fn open() -> (Memory, Arc<Dev>, String) {
    let card = 4 << 30;
    let uuid = format!("region{}-{}", std::process::id(), OPENS.fetch_add(1, SeqCst));
    let g = varena::GpuId { uuid: uuid.clone(), model: "host".into(), memory: card };
    assert!(varena::os::namespace(&format!("test{}-", std::process::id())));
    let ledger = varena::Ledger::create(&g, None).unwrap();
    std::mem::forget(ledger);
    let dev = Arc::new(Dev {
        host: HostBackend::new(card),
        instant: AtomicBool::new(true),
        next: AtomicU64::new(1),
        events: Mutex::new(HashMap::new()),
        landed: Condvar::new(),
        recorded: Mutex::new(Vec::new()),
        finished: Mutex::new(Vec::new()),
        waiting: Mutex::new(None),
    });
    let mem = Memory::open(g, None, Box::new(Shared(dev.clone()))).unwrap();
    mem.serve_standalone_torch().unwrap();
    (mem, dev, uuid)
}

fn until(what: &str, f: impl Fn() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < Duration::from_secs(20), "never: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn paint(ptr: u64, len: u64, seed: u8) {
    // SAFETY: a live block of `len` mapped host bytes the test allocated.
    let b = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len as usize) };
    b.iter_mut().enumerate().for_each(|(i, x)| *x = seed.wrapping_add(i as u8));
}

fn painted(ptr: u64, len: u64, seed: u8) -> bool {
    // SAFETY: a live block of `len` mapped host bytes the test allocated.
    let b = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    b.iter().enumerate().all(|(i, x)| *x == seed.wrapping_add(i as u8))
}

fn bytes(mem: &Memory, id: u64) -> u64 {
    mem.region_records().iter().find(|r| r.id == id).map_or(u64::MAX, |r| r.bytes)
}

/// A pinned region is Busy; its last unpin reclaims it on the reclaim thread; its next pin maps it
/// at the same address, fresh. Unpinned, a reclaim discards it at once.
#[test]
fn a_region_is_reclaimed_only_unpinned_and_comes_back_fresh() {
    let (mem, _dev, uuid) = open();
    let region = mem.region("kv", RegionKind::Kv, &[], false).unwrap();
    let p = {
        let filling = region.allocating().unwrap();
        assert!(!filling.fresh());
        mem.torch_alloc(3 * MB, A, None).unwrap()
    };
    paint(p, 3 * MB, 7);
    assert_eq!(bytes(&mem, region.id()), CHUNK);
    let pin = region.pin().unwrap();
    assert_eq!(mem.region_reclaim(region.id(), false).unwrap(), Reclaimed::Busy, "reclaimed while pinned");
    assert!(painted(p, 3 * MB, 7));
    drop(pin);
    until("the last unpin reclaims it", || bytes(&mem, region.id()) == 0);
    let pin = region.pin().unwrap();
    assert!(pin.fresh(), "a discarded region's next pin is fresh");
    assert_eq!(bytes(&mem, region.id()), CHUNK);
    paint(p, 3 * MB, 9);
    drop(pin);
    assert_eq!(mem.region_reclaim(region.id(), false).unwrap(), Reclaimed::Discarded(CHUNK));
    assert_eq!(mem.region_reclaim(region.id(), false).unwrap(), Reclaimed::Gone, "reclaimed twice");
    mem.torch_free(p, 3 * MB, A);
    let id = region.id();
    assert!(region.close().outlived.is_empty());
    assert_eq!(mem.region_reclaim(id, false).unwrap(), Reclaimed::Gone);
    drop(mem);
    varena::os::remove_table(&uuid);
}

/// A spilling region swaps: its next pin is not fresh and its bytes are back at their address.
#[test]
fn a_swapped_region_keeps_its_bytes() {
    let (mem, _dev, uuid) = open();
    let region = mem.region("latent", RegionKind::Latent, &[], true).unwrap();
    let blocks: Vec<(u64, u64)> = {
        let _filling = region.allocating().unwrap();
        [4096, 700_000, 9 * MB].iter().map(|&n| (mem.torch_alloc(n, A, None).unwrap(), n)).collect()
    };
    blocks.iter().enumerate().for_each(|(i, &(p, n))| paint(p, n, 31 * i as u8));
    let Reclaimed::Swapped(n) = mem.region_reclaim(region.id(), true).unwrap() else { panic!("not swapped") };
    assert!(n >= 2 * MB + CHUNK, "{n} bytes swapped");
    assert_eq!(bytes(&mem, region.id()), 0);
    let pin = region.pin().unwrap();
    assert!(!pin.fresh(), "a swapped region is not fresh");
    blocks.iter().enumerate().for_each(|(i, &(p, n))| assert!(painted(p, n, 31 * i as u8), "block {i} lost its bytes"));
    drop(pin);
    blocks.iter().for_each(|&(p, n)| mem.torch_free(p, n, A));
    drop(region);
    drop(mem);
    varena::os::remove_table(&uuid);
}

/// R2-3: a drop sets Dropping (acquire fails from then), then waits for the last released replay
/// of every stream the region replayed on, A and B both.
#[test]
fn a_graph_drop_waits_for_every_streams_last_replay() {
    let (mem, dev, uuid) = open();
    let graph = mem.graph_region(&[], &[], PoolId(1, 1)).unwrap();
    dev.instant.store(false, SeqCst);
    for stream in [A, A, B] {
        assert!(graph.acquire());
        graph.release(stream).unwrap();
    }
    let (a, b) = (dev.last(A).unwrap(), dev.last(B).unwrap());
    let (tx, rx) = mpsc::channel();
    let (m, id) = (mem.clone(), graph.id());
    std::thread::spawn(move || {
        let _ = tx.send(m.region_reclaim(id, false));
    });
    until("the drop waits on a replay", || dev.waiting.lock().is_some());
    assert!(graph.dropping() && !graph.acquire(), "a replay acquired while Dropping");
    let first = dev.waiting.lock().unwrap_or(A);
    dev.complete(first);
    let second = if first == A { B } else { A };
    until("the drop waits on the other stream", || *dev.waiting.lock() == Some(second));
    assert!(rx.try_recv().is_err(), "the drop returned before {second:#x}'s replay completed");
    dev.complete(second);
    assert_eq!(rx.recv_timeout(Duration::from_secs(20)).unwrap().unwrap(), Reclaimed::Dropped);
    let finished = dev.finished.lock().clone();
    assert!(finished.contains(&a) && finished.contains(&b), "finished {finished:?}, last replays {a} and {b}");
    assert_eq!(mem.region_reclaim(id, false).unwrap(), Reclaimed::Gone);
    graph.close();
    drop(mem);
    varena::os::remove_table(&uuid);
}

/// The verifier's HIGH: a release racing the drop. Whatever the interleaving, the drop finishes
/// the event of the last replay acquired before Dropping, never an older one.
#[test]
fn a_graph_drop_racing_a_replay_finishes_its_event() {
    let (mem, dev, uuid) = open();
    for round in 0..300u64 {
        let graph = Arc::new(mem.graph_region(&[], &[], PoolId(2, round)).unwrap());
        let replays = Arc::new(AtomicU64::new(0));
        let replayer = {
            let (graph, replays) = (graph.clone(), replays.clone());
            std::thread::spawn(move || {
                while graph.acquire() {
                    graph.release(G).unwrap();
                    replays.fetch_add(1, SeqCst);
                }
            })
        };
        let wait = round % 7;
        until("the replayer runs", || replays.load(SeqCst) >= wait);
        let before = dev.finished.lock().len();
        assert_eq!(mem.region_reclaim(graph.id(), false).unwrap(), Reclaimed::Dropped);
        replayer.join().unwrap();
        let finished: Vec<u64> = dev.finished.lock().iter().skip(before).copied().collect();
        if replays.load(SeqCst) > 0 {
            let last = dev.last(G).unwrap();
            assert!(finished.contains(&last), "round {round}: the drop finished {finished:?}, not the last replay's event {last}");
        }
        drop(graph);
    }
    drop(mem);
    varena::os::remove_table(&uuid);
}

/// Room: free is the device's free bytes less H plus the process's idle spans; an unpinned
/// region's bytes are cold, a pinned one's are not; no plan, no warm bytes or floor.
#[test]
fn room_counts_an_unpinned_region_cold() {
    let (mem, _dev, uuid) = open();
    let empty = mem.room(None);
    assert_eq!(empty.free, 4 << 30, "a host device's free bytes, no reserve");
    assert!(empty.granularity > 0);
    assert_eq!((empty.cold, empty.warm, empty.floor), (0, 0, 0));
    let region = mem.region("cond", RegionKind::Conditioning, &[], true).unwrap();
    let p = {
        let _filling = region.allocating().unwrap();
        mem.torch_alloc(5 * MB, A, None).unwrap()
    };
    assert_eq!(mem.room(None).cold, CHUNK, "an unpinned region is cold");
    let pin = region.pin().unwrap();
    assert_eq!(mem.room(None).cold, 0, "a pinned region is not cold");
    drop(pin);
    mem.torch_free(p, 5 * MB, A);
    drop(region);
    drop(mem);
    varena::os::remove_table(&uuid);
}
