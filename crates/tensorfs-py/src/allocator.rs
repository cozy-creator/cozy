//! torch's allocator asks varena for room itself: the segments of a pass's `torch.cuda.MemPool`
//! come from `varena_torch_alloc` (a `CUDAPluggableAllocator` over this library), and a shortfall
//! asks the thread's active request (`r.oom(need, site=None)`) and retries while varena answers
//! Retry. Null goes back only when it does not: torch then raises its OutOfMemoryError at the
//! call site, with no op ever replayed.
//!
//! It runs under torch's allocator lock (torch's Python bindings release the GIL before an op;
//! a thread that still holds it lets it go around varena's oom, which may wait on device work
//! whose host callbacks need it). Nothing here calls into torch or Python: varena's callbacks
//! (torch's trim, caches' shrinks) answer 0 on a thread inside it ([`inside`]), and the figures
//! varena gets carry no slack, so it asks for no trim.

use std::cell::{Cell, RefCell};
use std::ffi::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use tensorfs::io::cuda;
use varena::{Memory, Next, Request, SetId, Torch};

/// The pass a thread allocates for: its memory, request and set, and torch's bytes at its start
/// less the pool's then (the pool's own come from [`LIVE`]).
struct Active {
    memory: Arc<Memory>,
    request: Arc<Request>,
    set: Option<SetId>,
    outside: u64,
}

thread_local! {
    static ACTIVE: RefCell<Vec<Active>> = const { RefCell::new(Vec::new()) };
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// Bytes of the pool's live segments, segments made, shortfalls varena resolved (and their
/// bytes), allocations refused, and nanoseconds spent making segments.
static LIVE: AtomicU64 = AtomicU64::new(0);
static SEGMENTS: AtomicU64 = AtomicU64::new(0);
static RESOLVED: AtomicU64 = AtomicU64::new(0);
static RESOLVED_BYTES: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static NS: AtomicU64 = AtomicU64::new(0);
/// varena's callbacks into torch or Python reached on a thread inside the allocator (answered 0,
/// never called): trims, cache shrinks.
static TRIMS_INSIDE: AtomicU64 = AtomicU64::new(0);
static SHRINKS_INSIDE: AtomicU64 = AtomicU64::new(0);

/// Whether this thread is inside the allocator: varena's Python callbacks answer 0 here.
pub fn inside() -> bool {
    INSIDE.with(Cell::get)
}

/// A trim reached inside the allocator (it is answered 0); with slack 0 in the figures varena
/// gets from here it emits none.
pub fn trim_inside() {
    TRIMS_INSIDE.fetch_add(1, Relaxed);
}

/// A cache's shrink reached inside the allocator: answered 0, nothing freed under torch's lock.
pub fn shrink_inside() {
    SHRINKS_INSIDE.fetch_add(1, Relaxed);
}

/// (trims, shrinks) reached inside the allocator.
pub fn reentries() -> (u64, u64) {
    (TRIMS_INSIDE.load(Relaxed), SHRINKS_INSIDE.load(Relaxed))
}

/// The thread allocates for `request`'s pass of `set` (None: outside any pass) from now on.
pub fn enter(memory: Arc<Memory>, request: Arc<Request>, set: Option<SetId>, reserved: u64) {
    let outside = reserved.saturating_sub(LIVE.load(Relaxed));
    ACTIVE.with(|a| a.borrow_mut().push(Active { memory, request, set, outside }));
}

pub fn exit() {
    ACTIVE.with(|a| a.borrow_mut().pop());
}

/// (live bytes, segments, shortfalls resolved, their bytes, refusals, ns making segments).
pub fn stats() -> (u64, u64, u64, u64, u64, u64) {
    (LIVE.load(Relaxed), SEGMENTS.load(Relaxed), RESOLVED.load(Relaxed), RESOLVED_BYTES.load(Relaxed), REFUSED.load(Relaxed), NS.load(Relaxed))
}

fn segment(size: usize) -> Result<cuda::CUdeviceptr, cuda::CUresult> {
    let d = cuda::driver().map_err(|_| -1)?;
    let mut ptr: cuda::CUdeviceptr = 0;
    // SAFETY: the driver ABI with an out-param on our stack, in torch's current context.
    let rc = unsafe { (d.mem_alloc)(&mut ptr, size) };
    if rc == cuda::SUCCESS {
        Ok(ptr)
    } else {
        Err(rc)
    }
}

/// Whether a new segment of `size` keeps torch within what varena lets it hold now
/// (`torch_limit`); with no active pass, the driver alone decides.
fn fits(size: u64) -> bool {
    ACTIVE.with(|a| {
        a.borrow().last().is_none_or(|p| p.outside + LIVE.load(Relaxed) + size <= p.memory.torch_limit())
    })
}

extern "C" {
    /// CPython's own (stable since 3.4, outside the limited API the abi3 headers give): whether
    /// this thread holds the GIL, read without taking it.
    fn PyGILState_Check() -> c_int;
}

/// `f` without the GIL when this thread holds it: varena's oom may wait on device work (an
/// unmap's queued consumers), and a stream host callback in it may need the GIL. It never takes
/// the GIL: another thread may hold it while it waits for torch's allocator lock.
fn unlocked<T>(f: impl FnOnce() -> T) -> T {
    // SAFETY: the C API on this thread; the state saved is restored before returning.
    if unsafe { PyGILState_Check() } == 1 {
        let state = unsafe { pyo3::ffi::PyEval_SaveThread() };
        let out = f();
        unsafe { pyo3::ffi::PyEval_RestoreThread(state) };
        out
    } else {
        f()
    }
}

/// varena's answer to a shortfall of `need` bytes, for the thread's active pass.
fn ask(need: u64) -> Option<Next> {
    unlocked(|| ACTIVE.with(|a| {
        let active = a.borrow();
        let p = active.last()?;
        // Slack 0 (reserved == allocated): varena emits no trim, so torch's cache is never
        // emptied under its own lock; with nothing more to free it refuses, and torch then
        // releases its cached blocks itself and asks again.
        let held = p.outside + LIVE.load(Relaxed);
        let torch = Torch { allocated: held, reserved: held };
        Some(p.request.oom(p.set, Some(need), None, torch))
    }))
}

fn alloc(size: usize) -> *mut c_void {
    let began = Instant::now();
    INSIDE.with(|i| i.set(true));
    // A shortfall is varena's room (torch_limit) or the driver's: both ask varena, and a Retry
    // means it made progress (pages unmapped, the watermark lowered); anything else refuses.
    let ptr = loop {
        let got = if fits(size as u64) { segment(size) } else { Err(cuda::ERROR_OUT_OF_MEMORY) };
        match got {
            Ok(ptr) => break ptr as *mut c_void,
            Err(cuda::ERROR_OUT_OF_MEMORY) if matches!(ask(size as u64), Some(Next::Retry)) => {
                RESOLVED.fetch_add(1, Relaxed);
                RESOLVED_BYTES.fetch_add(size as u64, Relaxed);
            }
            Err(_) => {
                REFUSED.fetch_add(1, Relaxed);
                break std::ptr::null_mut();
            }
        }
    };
    INSIDE.with(|i| i.set(false));
    if !ptr.is_null() {
        LIVE.fetch_add(size as u64, Relaxed);
        SEGMENTS.fetch_add(1, Relaxed);
    }
    NS.fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    ptr
}

/// torch's pluggable allocator: a new segment of `size` bytes for the pool, room asked of varena
/// on a shortfall; null when varena cannot make it.
#[no_mangle]
pub extern "C" fn varena_torch_alloc(size: isize, _device: c_int, _stream: *mut c_void) -> *mut c_void {
    let size = usize::try_from(size).unwrap_or(0);
    catch_unwind(AssertUnwindSafe(|| alloc(size))).unwrap_or(std::ptr::null_mut())
}

/// The segment goes back to the driver.
#[no_mangle]
pub extern "C" fn varena_torch_free(ptr: *mut c_void, size: usize, _device: c_int, _stream: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(d) = cuda::driver() {
            // SAFETY: a segment `varena_torch_alloc` made; torch frees each once.
            if unsafe { (d.mem_free)(ptr as cuda::CUdeviceptr) } == cuda::SUCCESS {
                LIVE.fetch_sub(size as u64, Relaxed);
            }
        }
    }));
}
