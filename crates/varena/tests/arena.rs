//! torch's arenas over host pages (step 6 over leases): each review rule (R1, R2, R4/R5, R6, R10,
//! R11a), the no-learning-loop wait, fragmentation, and what a malloc+free pair costs. Events are
//! the test's own: an epoch lands only when the test completes its stream.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use tensorfs::io::Dst;
use varena::arena::{Chunk, Direct, Kind, Owner, PoolId, Source, Torch, CHUNK, FLOAT, OWN};
use varena::{Access, Backend, Event, HostBackend, Physical, Va};

const MB: u64 = 1 << 20;
const A: usize = 0xa00;
const B: usize = 0xb00;
const C: usize = 0xc00;

/// Host pages, with events that complete when the test says (or at once, `instant`). A query
/// waits while the test holds `stall`.
struct Gpu {
    host: HostBackend,
    events: Mutex<HashMap<u64, (usize, bool)>>,
    next: AtomicU64,
    instant: AtomicBool,
    records: AtomicUsize,
    queries: AtomicUsize,
    querying: AtomicBool,
    stall: Mutex<()>,
    /// Streams capturing now.
    capturing: Mutex<HashSet<usize>>,
    /// Pages this process created itself.
    creates: Arc<AtomicUsize>,
    /// Capture-status queries.
    looks: AtomicUsize,
}

impl Gpu {
    fn new(limit: u64, instant: bool) -> Arc<Gpu> {
        Arc::new(Gpu::value(limit, instant))
    }

    fn value(limit: u64, instant: bool) -> Gpu {
        Gpu {
            host: HostBackend::new(limit),
            events: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            instant: AtomicBool::new(instant),
            records: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            querying: AtomicBool::new(false),
            stall: Mutex::new(()),
            capturing: Mutex::new(HashSet::new()),
            creates: Arc::new(AtomicUsize::new(0)),
            looks: AtomicUsize::new(0),
        }
    }

    fn capture(&self, stream: usize, on: bool) {
        let mut c = self.capturing.lock();
        if on { c.insert(stream) } else { c.remove(&stream) };
    }

    /// Everything recorded on `stream` so far has run.
    fn complete(&self, stream: usize) {
        self.events.lock().values_mut().filter(|(s, _)| *s == stream).for_each(|(_, d)| *d = true);
    }
}

impl Backend for Gpu {
    fn capacity(&self) -> u64 { self.host.capacity() }
    fn free(&self) -> varena::Result<u64> { self.host.free() }
    fn process_used(&self) -> varena::Result<u64> { self.host.process_used() }
    fn opened_context(&self) -> u64 { 0 }
    fn granularity(&self) -> u64 { self.host.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.host.reserve(len) }
    fn release(&self, va: Va) { self.host.release(va) }
    fn create(&self, len: u64) -> varena::Result<Physical> {
        self.creates.fetch_add(1, SeqCst);
        self.host.create(len)
    }
    fn destroy(&self, page: Physical) -> varena::Result<()> { self.host.destroy(page) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> { self.host.map(va, offset, page, access) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> { self.host.protect(va, offset, len, access) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> { self.host.unmap(va, offset, len) }
    fn export(&self, page: &Physical) -> varena::Result<Owned> { self.host.export(page) }
    fn import(&self, handle: Owned, len: u64) -> varena::Result<Physical> { self.host.import(handle, len) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.host.target(va, offset, len) }
    fn copy_out(&self, va: Va, offset: u64, dst: &mut [u8], after: Event) -> varena::Result<()> { self.host.copy_out(va, offset, dst, after) }
    fn copy_in(&self, va: Va, offset: u64, src: &[u8]) -> varena::Result<()> { self.host.copy_in(va, offset, src) }
    fn fence(&self) -> varena::Result<Event> { Ok(Event::DONE) }
    fn record(&self, stream: usize) -> varena::Result<Event> {
        self.records.fetch_add(1, SeqCst);
        let id = self.next.fetch_add(1, SeqCst);
        self.events.lock().insert(id, (stream, self.instant.load(SeqCst)));
        Ok(Event(id))
    }
    fn query(&self, event: Event) -> varena::Result<bool> {
        self.querying.store(true, SeqCst);
        let _stalled = self.stall.lock();
        self.queries.fetch_add(1, SeqCst);
        Ok(event == Event::DONE || self.events.lock().get(&event.0).is_none_or(|e| e.1))
    }
    fn elapsed(&self, _: Event, _: Event) -> varena::Result<Option<u64>> { Ok(Some(0)) }
    fn wait(&self, _: Event, _: usize) -> varena::Result<()> { Ok(()) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> {
        self.complete(stream);
        Ok(())
    }
    fn finish(&self, _: Event) -> varena::Result<()> { Ok(()) }
    fn retire(&self, _: Event) {}
    fn drain(&self) -> varena::Result<()> { Ok(()) }
    fn capturing(&self, stream: usize) -> varena::Result<bool> {
        self.looks.fetch_add(1, SeqCst);
        Ok(self.capturing.lock().contains(&stream))
    }
}

/// [`Direct`], counted; while `closed`, a map waits for the test to open it.
struct Counted {
    inner: Direct,
    maps: AtomicUsize,
    unmaps: AtomicUsize,
    /// Every chunk mapped, in order.
    ids: Mutex<Vec<u64>>,
    /// Unmaps to refuse next, as a still-pinned chunk refuses.
    refuse: AtomicUsize,
    entered: AtomicBool,
    closed: Mutex<bool>,
    opened: Condvar,
}

impl Counted {
    fn open(&self) {
        *self.closed.lock() = false;
        self.opened.notify_all();
    }
}

impl Source for Counted {
    fn map(&self, va: Va, kind: Kind, into: &mut Vec<Chunk>) -> varena::Result<()> {
        self.entered.store(true, SeqCst);
        let mut closed = self.closed.lock();
        while *closed {
            self.opened.wait(&mut closed);
        }
        drop(closed);
        self.maps.fetch_add(1, SeqCst);
        let from = into.len();
        let mapped = self.inner.map(va, kind, into);
        self.ids.lock().extend(into.iter().skip(from).map(|c| c.id));
        mapped
    }

    fn unmap(&self, va: Va, chunks: Vec<Chunk>) -> Vec<(u64, Chunk)> {
        self.unmaps.fetch_add(1, SeqCst);
        if self.refuse.try_update(SeqCst, SeqCst, |n| n.checked_sub(1)).is_ok() {
            // Still pinned: nothing goes back.
            let mut at = va.base;
            return chunks.into_iter().map(|c| { let here = at; at += c.len; (here, c) }).collect();
        }
        self.inner.unmap(va, chunks)
    }
}

struct Rig {
    gpu: Arc<Gpu>,
    source: Arc<Counted>,
    torch: Arc<Torch>,
}

fn rig(limit: u64, instant: bool) -> Rig {
    let gpu = Gpu::new(limit, instant);
    let backend: Arc<dyn Backend> = gpu.clone();
    let source = Arc::new(Counted { inner: Direct::new(backend.clone(), limit), maps: AtomicUsize::new(0), unmaps: AtomicUsize::new(0), ids: Mutex::new(Vec::new()), refuse: AtomicUsize::new(0), entered: AtomicBool::new(false), closed: Mutex::new(false), opened: Condvar::new() });
    let torch = Torch::new(backend, source.clone()).unwrap();
    Rig { gpu, source, torch }
}

impl Rig {
    fn maps(&self) -> usize {
        self.source.maps.load(SeqCst)
    }
}

/// Waits for `what`, which the poller makes true; a test never sleeps a fixed time.
fn until(what: &str, f: impl Fn() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < Duration::from_secs(20), "never: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// `f` on another thread; it must return within a bound (a deadlock fails the test, not hangs it).
fn returns<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(20)).unwrap_or_else(|_| panic!("blocked: {what}"))
}

/// R1: a whole-free span serves its own stream again at once, but another stream (or the source)
/// only once an epoch its stream recorded after the free has completed.
#[test]
fn r1_a_span_leaves_its_stream_only_after_its_epoch() {
    let r = rig(512 * MB, false);
    let p = r.torch.alloc(4 * MB, A, None).unwrap();
    r.torch.free(p, 4 * MB, A);
    assert_eq!((r.torch.reset_peak(Owner::Process), r.torch.reset_peak(Owner::Process)), (4 * MB, 0), "a section's peak, then what is live");
    assert_eq!(r.torch.idle(), (0, CHUNK), "the whole-free span waits for A's epoch");
    assert_eq!(r.torch.alloc(4 * MB, A, None).unwrap(), p, "its own stream takes it back at once");
    r.torch.free(p, 4 * MB, A);
    let q = r.torch.alloc(4 * MB, B, None).unwrap();
    assert!(q.abs_diff(p) >= CHUNK, "stream B got A's span before A's epoch completed");
    assert_eq!(r.maps(), 2);
    assert_eq!(r.torch.reclaim(u64::MAX), 0, "the source got a span before its epoch");
    until("A's epoch is recorded", || r.gpu.records.load(SeqCst) > 0);
    until("A's epoch lands", || {
        r.gpu.complete(A);
        r.torch.idle().0 == CHUNK
    });
    assert_eq!(r.torch.alloc(4 * MB, C, None).unwrap(), p, "a quiet span serves any stream");
    assert_eq!(r.maps(), 2, "re-adopting an idle span maps nothing");
}

/// R1 for torch's record_stream: a block used on B frees on A only after B's epoch.
#[test]
fn r1_a_block_used_on_another_stream_frees_after_its_epoch() {
    let r = rig(512 * MB, false);
    let p = r.torch.alloc(4 * MB, A, None).unwrap();
    let keep = r.torch.alloc(4 * MB, A, None).unwrap();
    r.torch.record_stream(p, B);
    r.torch.free(p, 4 * MB, A);
    let other = r.torch.alloc(4 * MB, A, None).unwrap();
    assert_ne!(other, p, "a block B still reads was reused");
    r.torch.free(other, 4 * MB, A);
    assert_eq!(r.torch.stats().allocated, 8 * MB, "the deferred block counts until it frees");
    until("B's epoch is recorded", || r.gpu.records.load(SeqCst) > 0);
    until("the block frees", || {
        r.gpu.complete(B);
        r.torch.stats().allocated == 4 * MB
    });
    assert_eq!(r.torch.alloc(4 * MB, A, None).unwrap(), p, "best fit takes its exact hole back");
    r.torch.free(keep, 4 * MB, A);
}

/// R11a: the poller's driver calls never run under an arena lock, so a query stuck behind another
/// thread's driver work never stalls an allocation or a free.
#[test]
fn r11a_a_stalled_query_never_stalls_allocations() {
    let r = rig(512 * MB, false);
    let p = r.torch.alloc(4 * MB, A, None).unwrap();
    let stalled = r.gpu.stall.lock();
    r.torch.free(p, 4 * MB, A);
    until("the poller is inside a query", || r.gpu.querying.load(SeqCst));
    let torch = r.torch.clone();
    let done = returns("allocations beside a stalled query", move || {
        for i in 0..2000u64 {
            let (bytes, stream) = ([512, 70_000, 3 * MB, 80 * MB][(i % 4) as usize], [A, B][(i % 2) as usize]);
            let x = torch.alloc(bytes, stream, None).unwrap();
            torch.free(x, bytes, stream);
        }
        true
    });
    assert!(done);
    drop(stalled);
}

/// R2: while a capture is open, a non-pool block's free waits for the capture's end, and no epoch
/// is recorded on any stream; the capture's begin records those pending before it.
#[test]
fn r2_a_free_during_a_capture_waits_for_its_end() {
    let r = rig(512 * MB, true);
    let (p, keep) = (r.torch.alloc(4 * MB, A, None).unwrap(), r.torch.alloc(4 * MB, A, None).unwrap());
    let x = r.torch.alloc(4 * MB, B, None).unwrap();
    r.torch.free(x, 4 * MB, B);
    let pool = PoolId(1, 0);
    let (torch, gpu, (begun_tx, begun), (end_tx, end)) = (r.torch.clone(), r.gpu.clone(), mpsc::channel(), mpsc::channel::<()>());
    let capture = std::thread::spawn(move || {
        torch.pool_begin(pool, None);
        gpu.capture(C, true);
        begun_tx.send(()).unwrap();
        end.recv().unwrap();
        gpu.capture(C, false);
        torch.pool_end(pool);
    });
    begun.recv().unwrap();
    let records = r.gpu.records.load(SeqCst);
    assert!(records >= 1, "the begin recorded B's pending epoch");
    r.torch.free(p, 4 * MB, A);
    let other = r.torch.alloc(4 * MB, A, None).unwrap();
    r.torch.free(other, 4 * MB, A);
    assert_ne!(other, p, "a block freed during a capture was reused");
    let seen = r.gpu.queries.load(SeqCst);
    until("the poller runs during the capture", || r.gpu.queries.load(SeqCst) > seen + 2 || r.torch.idle().1 == 0);
    assert_eq!(r.gpu.records.load(SeqCst), records, "an epoch was recorded during a capture");
    assert_eq!(r.torch.stats().allocated, 12 * MB, "frees during the capture still count");
    end_tx.send(()).unwrap();
    capture.join().unwrap();
    until("the deferred free applies", || r.torch.stats().allocated == 4 * MB);
    assert_eq!(r.torch.alloc(4 * MB, A, None).unwrap(), p);
    r.torch.free(keep, 4 * MB, A);
}

/// R4/R5: a pool keeps every span while a graph may replay into it, and after its last release
/// until its fence (an event after the last replay) completes.
#[test]
fn r4_r5_a_pool_keeps_its_spans_until_released_and_fenced() {
    let r = rig(512 * MB, false);
    let pool = PoolId(7, 1);
    r.torch.pool_prepare(pool, 8 * MB).unwrap();
    let (torch, gpu) = (r.torch.clone(), r.gpu.clone());
    let blocks = returns("a capture", move || {
        torch.pool_begin(pool, None);
        gpu.capture(A, true);
        let mut got: Vec<u64> = (0..3).map(|_| torch.alloc(MB, A, Some(pool)).unwrap()).collect();
        let freed = got.pop().unwrap();
        torch.free(freed, MB, A);
        let again = torch.alloc(MB, A, Some(pool)).unwrap();
        gpu.capture(A, false);
        torch.pool_end(pool);
        assert_eq!(again, freed, "a free during capture returns to the pool");
        got.push(again);
        got
    });
    blocks.iter().for_each(|b| r.torch.free(*b, MB, A));
    assert_eq!((r.torch.idle(), r.torch.reclaim(u64::MAX)), ((0, 0), 0), "a pool span left while the pool is in use");
    r.torch.pool_fence(pool, C).unwrap();
    r.torch.pool_release(pool);
    let seen = r.gpu.queries.load(SeqCst);
    until("the poller queries the fence", || r.gpu.queries.load(SeqCst) > seen + 3);
    assert_eq!(r.torch.idle(), (0, 0), "a pool span left before its fence completed");
    until("the pool's span leaves once fenced", || {
        r.gpu.complete(C);
        r.torch.idle().0 == 8 * MB
    });
    assert_eq!(r.torch.stats().arenas.iter().find(|(o, _)| *o == Owner::Pool(pool)).map(|(_, s)| s.mapped), Some(0));
}

/// torch shares one pool across graphs: N begins, N releases; the spans leave at the last. A
/// pool never fenced waits for an epoch on every stream seen instead.
#[test]
fn a_shared_pool_leaves_at_its_last_release() {
    let r = rig(512 * MB, false);
    let pool = PoolId(9, 9);
    r.torch.pool_prepare(pool, 4 * MB).unwrap();
    let (torch, gpu) = (r.torch.clone(), r.gpu.clone());
    returns("two captures", move || {
        for _ in 0..2 {
            torch.pool_begin(pool, None);
            gpu.capture(A, true);
            let b = torch.alloc(MB, A, Some(pool)).unwrap();
            torch.free(b, MB, A);
            gpu.capture(A, false);
            torch.pool_end(pool);
        }
    });
    r.torch.pool_release(pool);
    assert_eq!(r.torch.idle(), (0, 0), "the pool left with one graph still holding it");
    r.torch.pool_release(pool);
    until("the fallback epoch is recorded", || r.gpu.records.load(SeqCst) > 0);
    assert_eq!(r.torch.idle(), (0, 0), "the pool left before its fallback epoch");
    until("the pool leaves", || {
        r.gpu.complete(A);
        r.torch.idle().0 == 4 * MB
    });
}

/// R6: a capturing thread, and anything routed to a pool, is served from mapped free bytes (its
/// arena's, or a quiet idle span's) or refused. It never reaches the source.
#[test]
fn r6_a_capturing_thread_never_reaches_the_source() {
    let r = rig(512 * MB, true);
    let small = r.torch.alloc(4096, A, None).unwrap();
    let spare = r.torch.alloc(40 * MB, B, None).unwrap();
    r.torch.free(spare, 40 * MB, B);
    until("B's span is quiet", || r.torch.idle().0 == 40 * MB);
    let maps = r.maps();
    let (torch, gpu) = (r.torch.clone(), r.gpu.clone());
    returns("a capture", move || {
        let pool = PoolId(3, 3);
        torch.pool_begin(pool, None);
        gpu.capture(A, true);
        let refused = torch.alloc(100 * MB, A, Some(pool));
        assert!(matches!(refused, Err(varena::Error::Exhausted(_))), "{refused:?}");
        let slot = torch.alloc(4096, A, None).unwrap();
        torch.free(slot, 4096, A);
        let adopted = torch.alloc(30 * MB, A, None).unwrap();
        let bigger = torch.alloc(300 * MB, A, None);
        assert!(matches!(bigger, Err(varena::Error::Exhausted(_))), "{bigger:?}");
        gpu.capture(A, false);
        torch.pool_end(pool);
        torch.free(adopted, 30 * MB, A);
    });
    assert_eq!(r.maps(), maps, "a capturing thread reached the source");
    r.torch.free(small, 4096, A);
}

/// R10: the source is called with no arena lock held: while one thread waits for its grant, frees,
/// allocations from mapped bytes and the agent's reclaim all go on.
#[test]
fn r10_a_grant_wait_holds_no_arena_lock() {
    let r = rig(512 * MB, true);
    let held = r.torch.alloc(MB, A, None).unwrap();
    let quiet = r.torch.alloc(40 * MB, B, None).unwrap();
    r.torch.free(quiet, 40 * MB, B);
    until("B's span is quiet", || r.torch.idle().0 > 0);
    *r.source.closed.lock() = true;
    let torch = r.torch.clone();
    let waiting = std::thread::spawn(move || torch.alloc(200 * MB, A, None));
    until("the allocation waits for its grant", || r.source.entered.load(SeqCst));
    let torch = r.torch.clone();
    let given = returns("work beside a grant wait", move || {
        torch.free(held, MB, A);
        let again = torch.alloc(MB, A, None).unwrap();
        torch.free(again, MB, A);
        let _ = torch.stats();
        torch.reclaim(u64::MAX)
    });
    assert!(given > 0, "the agent's reclaim took nothing");
    r.source.open();
    let p = waiting.join().unwrap().unwrap();
    r.torch.free(p, 200 * MB, A);
}

/// No learning loop: a shortfall waits for the grant (the granting thread never needs what the
/// waiter holds: a stand-in for the GIL), and gives this process's idle spans back before it is
/// refused, waiting on their epochs if it must. It is refused only at the floor.
#[test]
fn a_shortfall_waits_for_a_grant_and_is_refused_only_at_the_floor() {
    let r = rig(128 * MB, false);
    let gil = Arc::new(Mutex::new(()));
    *r.source.closed.lock() = true;
    let (torch, g) = (r.torch.clone(), gil.clone());
    let waiter = std::thread::spawn(move || {
        let _gil = g.lock();
        torch.alloc(30 * MB, A, None)
    });
    until("the waiter asks", || r.source.entered.load(SeqCst));
    r.source.open();
    let p = waiter.join().unwrap().unwrap();
    r.torch.free(p, 30 * MB, A);
    // 32 MiB idle on A, its epoch pending; 96 MiB left: a 100 MiB block needs A's span back.
    until("A's epoch is recorded", || r.gpu.records.load(SeqCst) > 0);
    let (gpu, source, torch) = (r.gpu.clone(), r.source.clone(), r.torch.clone());
    let lands = std::thread::spawn(move || {
        until("the big allocation asks", || source.maps.load(SeqCst) >= 2);
        until("A's span goes back", || {
            gpu.complete(A);
            torch.idle() == (0, 0)
        });
    });
    let big = r.torch.alloc(100 * MB, B, None).unwrap();
    lands.join().unwrap();
    assert_eq!(r.torch.idle(), (0, 0), "the idle span went back to the source");
    let refused = r.torch.alloc(30 * MB, B, None);
    match refused {
        Err(varena::Error::Exhausted(e)) => assert!(e.site.as_deref().unwrap_or_default().contains("budget"), "{e:?}"),
        other => panic!("expected the floor's refusal, got {other:?}"),
    }
    r.torch.free(big, 100 * MB, B);
}

/// A capture whose begin threw gets no end from torch. Once every stream it allocated on has
/// stopped capturing, the poller ends it: frees deferred by it apply, and a shortfall that waits
/// on epochs completes (no hang). One that never allocated ends at the Runtime's pool_end.
#[test]
fn a_capture_that_died_without_an_end_is_reaped() {
    let r = rig(512 * MB, false);
    let pool = PoolId(4, 4);
    r.torch.pool_prepare(pool, 2 * MB).unwrap();
    let (p, keep) = (r.torch.alloc(4 * MB, B, None).unwrap(), r.torch.alloc(4 * MB, B, None).unwrap());
    let (torch, gpu) = (r.torch.clone(), r.gpu.clone());
    returns("a capture that dies", move || {
        torch.pool_begin(pool, None);
        gpu.capture(A, true);
        torch.alloc(MB, A, Some(pool)).unwrap();
        // The capture stops (its thread died mid-capture): no pool_end.
        gpu.capture(A, false);
    });
    r.torch.free(p, 4 * MB, B);
    until("the dead capture is reaped and the free applies", || {
        r.gpu.complete(B);
        r.torch.stats().allocated == 4 * MB + MB
    });
    r.torch.free(keep, 4 * MB, B);
    let torch = r.torch.clone();
    returns("a capture that never allocated", move || {
        torch.pool_begin(PoolId(5, 5), None);
        torch.pool_end(PoolId(5, 5));
        torch.pool_release(PoolId(5, 5));
        let x = torch.alloc(4 * MB, C, None).unwrap();
        torch.free(x, 4 * MB, C);
    });
    until("frees apply again", || r.torch.stats().allocated == MB);
}

/// A shortfall beside another thread's open capture still completes: the capture's begin recorded
/// the pending epochs, and the poller queries them while the capture stays open.
#[test]
fn a_shortfall_beside_an_open_capture_completes() {
    let r = rig(128 * MB, false);
    let p = r.torch.alloc(30 * MB, A, None).unwrap();
    r.torch.free(p, 30 * MB, A);
    let (torch, gpu, (open_tx, open), (end_tx, end)) = (r.torch.clone(), r.gpu.clone(), mpsc::channel(), mpsc::channel::<()>());
    let capture = std::thread::spawn(move || {
        torch.pool_begin(PoolId(6, 6), None);
        gpu.capture(B, true);
        open_tx.send(()).unwrap();
        end.recv().unwrap();
        gpu.capture(B, false);
        torch.pool_end(PoolId(6, 6));
    });
    open.recv().unwrap();
    let (gpu, source) = (r.gpu.clone(), r.source.clone());
    let lands = std::thread::spawn(move || {
        until("the big allocation asks", || source.maps.load(SeqCst) >= 2);
        gpu.complete(A);
    });
    let big = r.torch.alloc(100 * MB, C, None).unwrap();
    lands.join().unwrap();
    assert_eq!(r.torch.idle(), (0, 0), "A's idle span went back while the capture stayed open");
    end_tx.send(()).unwrap();
    capture.join().unwrap();
    r.torch.free(big, 100 * MB, C);
}

/// A begin that threw (cudaStreamBeginCapture failed): its stream never captures. The poller
/// leaves it (right after the hook, a capture about to start looks the same). With its stream
/// known, the thread's next allocator call ends it; with none known, torch's reset or the
/// Runtime's capture_failed does, from any thread.
#[test]
fn a_begin_that_threw_is_ended() {
    let r = rig(512 * MB, true);
    let (p, keep) = (r.torch.alloc(4 * MB, B, None).unwrap(), r.torch.alloc(4 * MB, B, None).unwrap());
    let (torch, (began_tx, began), (next_tx, next)) = (r.torch.clone(), mpsc::channel(), mpsc::channel::<()>());
    let threw = std::thread::spawn(move || {
        torch.pool_begin(PoolId(8, 8), Some(A));
        began_tx.send(()).unwrap();
        next.recv().unwrap();
        let x = torch.alloc(4096, C, None).unwrap();
        torch.free(x, 4096, C);
    });
    began.recv().unwrap();
    r.torch.free(p, 4 * MB, B);
    let looks = r.gpu.looks.load(SeqCst);
    until("the poller looks at the open capture", || r.gpu.looks.load(SeqCst) > looks + 3);
    assert_eq!(r.torch.stats().allocated, 8 * MB, "the poller ended a capture never seen");
    next_tx.send(()).unwrap();
    threw.join().unwrap();
    until("the thread's next call ended it", || r.torch.stats().allocated == 4 * MB);
    let torch = r.torch.clone();
    std::thread::spawn(move || torch.pool_begin(PoolId(9, 1), None)).join().unwrap();
    r.torch.free(keep, 4 * MB, B);
    assert_eq!(r.torch.stats().allocated, 4 * MB, "a free during the dead capture applied");
    r.torch.pool_end(PoolId(9, 1));
    until("torch's reset on another thread ended it", || r.torch.stats().allocated == 0);
}

/// A chunk the source cannot take back yet (still pinned, or its unmap failed) stays mapped: its
/// addresses are never handed out again and it stays counted, and the next give-back pass returns
/// it.
#[test]
fn a_chunk_that_cannot_go_back_keeps_its_addresses_until_it_does() {
    let r = rig(512 * MB, true);
    let p = r.torch.alloc(40 * MB, A, None).unwrap();
    r.torch.free(p, 40 * MB, A);
    until("the span is quiet", || r.torch.idle().0 == 40 * MB);
    r.source.refuse.store(1, SeqCst);
    assert_eq!(r.torch.reclaim(u64::MAX), 0, "a refused chunk counted as given back");
    assert_eq!((r.torch.idle(), r.torch.stats().mapped), ((0, 0), 40 * MB), "the stuck span is no longer mapped in the books");
    let others: Vec<u64> = (0..3).map(|_| r.torch.alloc(40 * MB, B, None).unwrap()).collect();
    assert!(others.iter().all(|o| o + 40 * MB <= p || *o >= p + 40 * MB), "a stuck chunk's addresses were handed out: {others:x?} over {p:#x}");
    assert_eq!(r.torch.stats().mapped, 160 * MB);
    let unmaps = r.source.unmaps.load(SeqCst);
    assert_eq!(r.torch.reclaim(0), 40 * MB, "the next give-back pass did not return it");
    assert!(r.source.unmaps.load(SeqCst) > unmaps);
    assert_eq!(r.torch.stats().mapped, 120 * MB);
    assert_eq!(r.torch.reclaim(0), 0, "a returned chunk was retried again");
    others.iter().for_each(|o| r.torch.free(*o, 40 * MB, B));
}

/// A pass's peak covers the sections measured inside it; each section's peak is its own window.
#[test]
fn a_pass_peak_covers_its_sections() {
    let r = rig(512 * MB, true);
    let route = r.torch.route(Owner::Request(4));
    let req = Owner::Request(4);
    r.torch.pass_start(req);
    let big = r.torch.alloc(10 * MB, A, None).unwrap();
    r.torch.free(big, 10 * MB, A);
    assert_eq!(r.torch.reset_peak(req), 10 * MB, "the first section");
    let small = r.torch.alloc(4 * MB, A, None).unwrap();
    assert_eq!(r.torch.reset_peak(req), 4 * MB, "the second section is its own window");
    r.torch.free(small, 4 * MB, A);
    assert_eq!(r.torch.pass_peak(req), 10 * MB, "the pass's peak lost a section's");
    r.torch.pass_start(req);
    assert_eq!(r.torch.pass_peak(req), 0, "the next pass starts at what is live");
    drop(route);
}

/// The manager asks by chunk: exactly the asked idle chunk goes back, its spare split around
/// it, and the rest stays idle for the arenas.
#[test]
fn an_asked_chunk_goes_back_alone() {
    let r = rig(512 * MB, true);
    let big = r.torch.alloc(70 * MB, A, None).unwrap();
    r.torch.free(big, 70 * MB, A);
    until("the span is quiet", || r.torch.idle().0 == 70 * MB);
    let asked = r.source.ids.lock()[1];
    assert_eq!(r.torch.reclaim_chunks(&[asked]), CHUNK);
    assert_eq!((r.torch.idle().0, r.source.unmaps.load(SeqCst)), (70 * MB - CHUNK, 1));
    assert_eq!(r.torch.reclaim_chunks(&[asked]), 0, "a chunk given back twice");
    let maps = r.maps();
    let again = r.torch.alloc(30 * MB, A, None).unwrap();
    assert_eq!((again, r.maps()), (big, maps), "the first chunk stayed idle and serves again");
}

/// A tiny xorshift: the same sequence every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Mostly small blocks, some medium and large, a few of their own span.
    fn size(&mut self) -> u64 {
        match self.next() % 100 {
            0..=59 => 1 + self.next() % (64 << 10),
            60..=84 => (64 << 10) + self.next() % MB,
            85..=97 => MB + self.next() % (40 * MB),
            _ => OWN + self.next() % (64 * MB),
        }
    }
}

/// Live blocks never overlap.
fn place(live: &mut BTreeMap<u64, u64>, at: u64, len: u64) {
    if let Some((&b, &l)) = live.range(..at + len).next_back() {
        assert!(b + l <= at, "{at:#x}+{len} overlaps {b:#x}+{l}");
    }
    live.insert(at, len);
}

/// One round of mixed blocks on one stream, freed in a shuffled order but `keep`.
fn round(r: &Rig, seed: u64, keep: usize) -> Vec<(u64, u64)> {
    let mut rng = Rng(seed);
    let mut live = BTreeMap::new();
    let mut blocks: Vec<(u64, u64)> = (0..600)
        .map(|_| {
            let size = rng.size();
            let at = r.torch.alloc(size, A, None).unwrap();
            place(&mut live, at, size.max(1).next_multiple_of(512));
            (at, size)
        })
        .collect();
    for i in (1..blocks.len()).rev() {
        blocks.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let kept = blocks.split_off(blocks.len() - keep);
    blocks.iter().for_each(|(at, size)| r.torch.free(*at, *size, A));
    kept
}

/// One request: a round in its own arena, then its end.
fn request(r: &Rig, id: u64, seed: u64, keep: usize) -> (Vec<(u64, u64)>, varena::arena::Ended) {
    let route = r.torch.route(Owner::Request(id));
    let kept = round(r, seed, keep);
    drop(route);
    (kept, r.torch.end(Owner::Request(id)))
}

/// Fragmentation: a request's spans all come free whole, whatever its order of frees. Within the
/// request they serve its next round with no new mapping; past its end, all but the float go
/// back to the manager (leases rev 6: no idle hoarding), and what is mapped is the float alone.
#[test]
fn a_request_comes_free_whole_and_gives_back_all_but_the_float() {
    let r = rig(8 << 30, true);
    let route = r.torch.route(Owner::Request(1));
    round(&r, 0x9e3779b97f4a7c15, 0);
    let maps = r.maps();
    round(&r, 0x9e3779b97f4a7c15, 0);
    assert_eq!(r.maps(), maps, "the request's second round mapped new spans");
    drop(route);
    let ended = r.torch.end(Owner::Request(1));
    assert!(ended.outlived.is_empty(), "{:?}", ended.outlived);
    until("idle trims to the float", || {
        let s = r.torch.stats();
        s.idle == s.mapped && s.mapped <= 2 * OWN && r.torch.idle().1 == 0
    });
    let s = r.torch.stats();
    assert!(r.source.unmaps.load(SeqCst) > 0, "nothing went back");
    assert_eq!((s.allocated, s.fragmentation), (0, s.mapped));
    assert!(s.arenas.iter().all(|(o, _)| *o == Owner::Process), "the ended arena stayed registered");
}

/// Idle beyond the float goes back as soon as its epoch passed; fragmentation is what is mapped
/// beyond the live bytes.
#[test]
fn idle_beyond_the_float_goes_back_promptly() {
    let r = rig(512 * MB, true);
    let live = r.torch.alloc(MB, B, None).unwrap();
    let blocks: Vec<u64> = (0..3).map(|_| r.torch.alloc(40 * MB, A, None).unwrap()).collect();
    blocks.iter().for_each(|b| r.torch.free(*b, 40 * MB, A));
    until("idle trims to the float", || r.torch.idle().0 <= FLOAT && r.source.unmaps.load(SeqCst) >= 2);
    let s = r.torch.stats();
    assert_eq!(s.fragmentation, s.mapped - s.allocated);
    assert_eq!(s.allocated, MB);
    assert!(s.mapped <= FLOAT + 2 * MB, "{} mapped beyond the float and the live span", s.mapped);
    r.torch.free(live, MB, B);
}

/// The request-end audit names a survivor; its span stays until it frees, then leaves too.
#[test]
fn a_survivor_is_named_and_its_span_leaves_once_it_frees() {
    let r = rig(8 << 30, true);
    let (kept, ended) = request(&r, 3, 12345, 1);
    let [(ptr, size)] = kept[..] else { unreachable!() };
    assert_eq!(ended.outlived.len(), 1);
    assert_eq!((ended.outlived[0].ptr, ended.outlived[0].stream), (ptr, A));
    assert!(ended.outlived[0].bytes >= size);
    let s = r.torch.stats();
    assert!(s.idle < s.mapped, "the survivor's span left");
    r.torch.free(ptr, size, A);
    until("everything mapped is idle", || {
        let s = r.torch.stats();
        (s.allocated, s.idle) == (0, s.mapped)
    });
}

/// M2/R11: a cache region's blocks never share a span with a request's, so the region's spans
/// come free whole while the request's stay.
#[test]
fn a_region_has_spans_of_its_own() {
    let r = rig(512 * MB, true);
    let region = r.torch.route(Owner::Region(5));
    let cached: Vec<u64> = (0..8).map(|_| r.torch.alloc(4096, A, None).unwrap()).collect();
    drop(region);
    let req = r.torch.route(Owner::Request(6));
    let live = r.torch.alloc(4096, A, None).unwrap();
    let mapped = |o: Owner| r.torch.stats().arenas.iter().find(|(x, _)| *x == o).map_or(0, |(_, s)| s.mapped);
    assert!(mapped(Owner::Region(5)) > 0 && mapped(Owner::Request(6)) > 0);
    cached.iter().for_each(|c| r.torch.free(*c, 4096, A));
    assert_eq!(mapped(Owner::Region(5)), 0, "the region's span did not come free whole");
    assert!(mapped(Owner::Request(6)) > 0);
    assert!(r.torch.end(Owner::Region(5)).outlived.is_empty());
    r.torch.free(live, 4096, A);
    drop(req);
}

/// Fills `len` bytes at `ptr` (host pages) with a pattern of `seed`.
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

fn mapped(r: &Rig, o: Owner) -> u64 {
    r.torch.stats().arenas.iter().find(|(x, _)| *x == o).map_or(0, |(_, s)| s.mapped)
}

/// §5 Evict/MakeResident: a swapped region's chunks go back to the source, and come again at the
/// same addresses with its bytes; its blocks' pointers never change.
#[test]
fn a_swapped_region_comes_back_at_its_addresses_with_its_bytes() {
    let r = rig(512 * MB, true);
    let route = r.torch.route(Owner::Region(11));
    let blocks: Vec<(u64, u64)> = [512, 40_000, 3 * MB, 80 * MB].iter().map(|&n| (r.torch.alloc(n, A, None).unwrap(), n)).collect();
    drop(route);
    blocks.iter().enumerate().for_each(|(i, &(p, n))| paint(p, n, i as u8));
    let before = (mapped(&r, Owner::Region(11)), r.torch.stats().mapped, r.source.unmaps.load(SeqCst));
    let mut ev = r.torch.evict(Owner::Region(11), true).unwrap().expect("no capture is open");
    assert!(ev.swapped());
    assert_eq!(ev.bytes, before.0, "every span of the region was evicted");
    assert_eq!(mapped(&r, Owner::Region(11)), 0);
    assert_eq!(r.torch.stats().mapped, before.1 - before.0, "the device's mapped bytes dropped once");
    assert!(r.source.unmaps.load(SeqCst) > before.2, "its chunks went back to the source");
    assert!(!r.torch.make_resident(Owner::Region(11), &mut ev).unwrap(), "swapped bytes are not fresh");
    assert_eq!((mapped(&r, Owner::Region(11)), r.torch.stats().mapped), (before.0, before.1));
    blocks.iter().enumerate().for_each(|(i, &(p, n))| assert!(painted(p, n, i as u8), "block {i} lost its bytes"));
    let route = r.torch.route(Owner::Region(11));
    let more = r.torch.alloc(512, A, None).unwrap();
    drop(route);
    assert!(blocks.iter().all(|&(p, n)| more >= p + n || more + 512 <= p), "a resident region's free bytes serve again");
    blocks.iter().for_each(|&(p, n)| r.torch.free(p, n, A));
    r.torch.free(more, 512, A);
    assert!(r.torch.end(Owner::Region(11)).outlived.is_empty());
}

/// An evicted span is never carved: its free runs and slot pages stay out of the free lists even
/// as its blocks free. One that comes whole free waits for the next residency, or gives its
/// addresses back once its arena ended.
#[test]
fn an_evicted_span_is_never_carved_from() {
    let r = rig(512 * MB, true);
    let route = r.torch.route(Owner::Region(12));
    let small: Vec<u64> = (0..4).map(|_| r.torch.alloc(4096, A, None).unwrap()).collect();
    let run = r.torch.alloc(5 * MB, A, None).unwrap();
    let mut ev = r.torch.evict(Owner::Region(12), false).unwrap().unwrap();
    assert!(!ev.swapped());
    r.torch.free(small[0], 4096, A);
    r.torch.free(run, 5 * MB, A);
    // The region's first blocks start its spans: a 2 MiB one for the slots, a chunk for the run.
    let (slots, runs) = (small[0]..small[0] + 2 * MB, run..run + CHUNK);
    let fresh: Vec<u64> = (0..3).map(|_| r.torch.alloc(4096, A, None).unwrap()).chain([r.torch.alloc(5 * MB, A, None).unwrap()]).collect();
    for p in &fresh {
        assert!(!slots.contains(p) && !runs.contains(p), "{p:#x} was carved from an evicted span");
    }
    assert!(r.torch.make_resident(Owner::Region(12), &mut ev).unwrap(), "discarded bytes are fresh");
    let again = r.torch.alloc(4096, A, None).unwrap();
    assert!(slots.contains(&again), "the resident span serves again");
    drop(route);
    for p in small.iter().skip(1).chain([&again]).chain(fresh.iter().take(3)) {
        r.torch.free(*p, 4096, A);
    }
    if let Some(p) = fresh.get(3) {
        r.torch.free(*p, 5 * MB, A);
    }
    assert_eq!(mapped(&r, Owner::Region(12)), 0);
    // Evicted, whole free, then ended: its addresses go back and nothing stays mapped.
    let route = r.torch.route(Owner::Region(13));
    let p = r.torch.alloc(4096, A, None).unwrap();
    drop(route);
    let _gone = r.torch.evict(Owner::Region(13), false).unwrap().unwrap();
    r.torch.free(p, 4096, A);
    r.torch.forget_evicted(Owner::Region(13));
    assert!(r.torch.end(Owner::Region(13)).outlived.is_empty());
    assert_eq!(r.torch.stats().arenas.iter().filter(|(o, _)| *o == Owner::Region(13)).count(), 0);
}

/// R1 for an eviction: it waits for an epoch on the region's stream after everything enqueued
/// so far, with no arena lock held (another stream allocates meanwhile); a capture open refuses it.
#[test]
fn an_eviction_waits_for_the_regions_stream_and_never_during_a_capture() {
    let r = Arc::new(rig(512 * MB, false));
    let route = r.torch.route(Owner::Region(14));
    let p = r.torch.alloc(3 * MB, A, None).unwrap();
    drop(route);
    r.torch.pool_begin(PoolId(9, 9), Some(C));
    assert!(r.torch.evict(Owner::Region(14), false).unwrap().is_none(), "evicted while a capture is open");
    r.torch.pool_end(PoolId(9, 9));
    let (tx, rx) = mpsc::channel();
    let r2 = r.clone();
    std::thread::spawn(move || {
        let _ = tx.send(r2.torch.evict(Owner::Region(14), false).map(|e| e.map(|e| e.bytes)));
    });
    until("A's epoch is recorded", || r.gpu.events.lock().values().any(|(s, _)| *s == A));
    let other = returns("an allocation beside a waiting eviction", {
        let r = r.clone();
        move || r.torch.alloc(4096, B, None).unwrap()
    });
    assert!(rx.try_recv().is_err(), "the eviction did not wait for A's epoch");
    assert_eq!(mapped(&r, Owner::Region(14)), CHUNK);
    r.gpu.complete(A);
    let got = rx.recv_timeout(Duration::from_secs(20)).expect("the eviction never returned").unwrap();
    assert_eq!(got, Some(CHUNK));
    r.torch.free(other, 4096, B);
    r.torch.free(p, 3 * MB, A);
}

/// Through Memory, as S3's shim calls it: nothing is served until a source is chosen; then a
/// request's guard routes its thread (dropped on another thread, it stops), and the request's
/// end record carries the audit.
#[test]
fn memory_routes_a_request_and_audits_its_end() {
    let card = 4 << 30;
    let g = varena::GpuId { uuid: format!("arena{}", std::process::id()), model: "host".into(), memory: card };
    assert!(varena::os::namespace(&format!("test{}-", std::process::id())));
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let gpu = Gpu::value(card, true);
    let creates = gpu.creates.clone();
    let mem = varena::Memory::open(g.clone(), None, Box::new(gpu)).unwrap();
    let refused = mem.torch_alloc(4096, A, None);
    assert!(matches!(refused, Err(varena::Error::Invalid(_))), "{refused:?}");
    assert_eq!(creates.load(SeqCst), 0, "a page was made before torch was served");
    mem.serve_standalone_torch().unwrap();
    let r = mem.request("audit", varena::Plan::default()).unwrap();
    let guard = mem.allocating(&r);
    let kept = mem.torch_alloc(5000, A, None).unwrap();
    let freed = mem.torch_alloc(3 * MB, A, None).unwrap();
    mem.torch_free(freed, 3 * MB, A);
    assert_eq!(mem.torch_stats().requests.len(), 1);
    std::thread::spawn(move || drop(guard)).join().unwrap();
    let process = mem.torch_alloc(4096, A, None).unwrap();
    let stats = r.end();
    assert_eq!(stats.outlived.iter().map(|o| o.ptr).collect::<Vec<_>>(), vec![kept], "the audit");
    mem.torch_free(kept, 5000, A);
    mem.torch_free(process, 4096, A);
    drop(mem);
    varena::os::remove_table(&g.uuid);
}

/// A leased Memory against a real manager in this process: torch is refused until served (no page
/// made outside the books); then its spans are the manager's grants, it is refused only at the
/// manager's floor, and what it gives back leaves the books.
#[cfg(target_os = "linux")]
#[test]
fn a_leased_memory_allocates_only_from_its_grants() {
    use std::os::unix::net::{UnixListener, UnixStream};
    use varena::lease::{Hosting, Identity, Manager};

    struct Me;
    impl Hosting for Me {
        fn identify(&self, peer: &platform::handles::Peer) -> Option<Identity> {
            (peer.pid == std::process::id()).then(|| Identity { executor: "me".into(), domain: "test".into() })
        }
    }

    let (pid, card) = (std::process::id(), 256 * MB);
    assert!(varena::os::namespace(&format!("test{pid}-")));
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("lease.sock");
    let host = varena::GpuId { uuid: format!("arena-host{pid}"), model: "host".into(), memory: card };
    let open = Box::new(move || Ok(Arc::new(HostBackend::new(card)) as Arc<dyn Backend>));
    let manager = Manager::create(&host, Some(card), varena::ledger::Mode::Dedicated, open, UnixListener::bind(&socket).unwrap(), Arc::new(Me)).unwrap();
    varena::os::remove_table(&host.uuid);
    let g = varena::GpuId { uuid: format!("arena-leased{pid}"), model: "host".into(), memory: card };
    let _ledger = varena::Ledger::create(&g, None).unwrap();
    let gpu = Gpu::value(card, true);
    let creates = gpu.creates.clone();
    let mem = varena::Memory::open(g.clone(), None, Box::new(gpu)).unwrap();
    // Leasing serves torch from the manager's grants at once: no page made outside the books.
    assert_eq!(mem.lease_over(UnixStream::connect(&socket).unwrap()).unwrap(), None, "the manager refused the lease");
    assert_eq!((manager.holdings().leased, creates.load(SeqCst)), (0, 0), "memory outside the books");
    let p = mem.torch_alloc(4 * MB, A, None).unwrap();
    until("the span is the manager's grant", || manager.holdings().leased == CHUNK);
    let floor = mem.torch_alloc(300 * MB, A, None);
    assert!(matches!(floor, Err(varena::Error::Exhausted(_))), "{floor:?}");
    mem.torch_free(p, 4 * MB, A);
    let torch = mem.torch().unwrap();
    until("the span is quiet", || torch.idle().0 == CHUNK);
    assert_eq!(torch.reclaim(u64::MAX), CHUNK);
    until("the chunk leaves the books", || manager.holdings().leased == 0);
    assert_eq!(creates.load(SeqCst), 0, "the executor made a page itself");
    drop((torch, mem));
    varena::os::remove_table(&g.uuid);
}

/// `pairs` malloc+free pairs of `sizes` on `stream`, 32 to 64 blocks live: ns per pair.
fn churn(torch: &Torch, sizes: &[u64], stream: usize, pairs: usize) -> f64 {
    let _step = torch.route(Owner::Request(stream as u64));
    let mut held: Vec<(u64, u64)> = Vec::with_capacity(64);
    let start = Instant::now();
    for i in 0..pairs {
        let size = sizes[i % sizes.len()];
        held.push((torch.alloc(size, stream, None).unwrap(), size));
        if held.len() == 64 {
            held.drain(..32).for_each(|(p, s)| torch.free(p, s, stream));
        }
    }
    held.drain(..).for_each(|(p, s)| torch.free(p, s, stream));
    start.elapsed().as_nanos() as f64 / pairs as f64
}

/// What a malloc+free pair costs, single stream and two threads on two streams.
#[test]
#[ignore = "a measurement: run alone, with --nocapture"]
fn malloc_free_cost() {
    const PAIRS: usize = 1_000_000;
    let r = rig(16 << 30, true);
    let mut rng = Rng(77);
    let sizes: Arc<Vec<u64>> = Arc::new((0..1024).map(|_| match rng.next() % 10 { 0..=5 => 1 + rng.next() % (64 << 10), 6..=8 => (64 << 10) + rng.next() % (8 * MB), _ => 8 * MB + rng.next() % (40 * MB) }).collect());
    churn(&r.torch, &sizes, A, PAIRS);
    let one = churn(&r.torch, &sizes, A, PAIRS);
    let spawn = |stream| {
        let (torch, sizes) = (r.torch.clone(), sizes.clone());
        std::thread::spawn(move || churn(&torch, &sizes, stream, PAIRS))
    };
    let (a, b) = (spawn(A), spawn(B));
    let (a, b) = (a.join().unwrap(), b.join().unwrap());
    println!("malloc+free pair: {one:.0} ns on one stream; {a:.0} / {b:.0} ns per pair with two threads on two streams");
}
