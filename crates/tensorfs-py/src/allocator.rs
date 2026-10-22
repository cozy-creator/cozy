//! torch's allocator asks varena for room itself: the segments of a pass's `torch.cuda.MemPool`
//! come from `varena_torch_alloc` (a `CUDAPluggableAllocator` over this library), and a shortfall
//! asks the thread's active request (`r.oom(need, site=None)`) and retries while varena answers
//! Retry. Null goes back only when it does not: torch then raises its OutOfMemoryError at the
//! call site, with no op ever replayed.
//!
//! It runs under torch's allocator lock, with the GIL held: nothing here calls into torch or
//! Python. varena's callbacks into Python (torch's trim, caches' shrinks) answer 0 on a thread
//! inside it ([`inside`]); on another thread they wait for the GIL, and nothing here waits for
//! them (varena's answer never waits on an ask).

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
    allocated: u64,
    outside: u64,
}

thread_local! {
    static ACTIVE: RefCell<Vec<Active>> = const { RefCell::new(Vec::new()) };
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// Bytes of the pool's live segments, segments made, shortfalls varena resolved, allocations
/// refused, and nanoseconds spent making segments.
static LIVE: AtomicU64 = AtomicU64::new(0);
static SEGMENTS: AtomicU64 = AtomicU64::new(0);
static RESOLVED: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static NS: AtomicU64 = AtomicU64::new(0);

/// Whether this thread is inside the allocator: varena's Python callbacks answer 0 here.
pub fn inside() -> bool {
    INSIDE.with(Cell::get)
}

/// The thread allocates for `request`'s pass of `set` (None: outside any pass) from now on.
pub fn enter(memory: Arc<Memory>, request: Arc<Request>, set: Option<SetId>, allocated: u64, reserved: u64) {
    let outside = reserved.saturating_sub(LIVE.load(Relaxed));
    ACTIVE.with(|a| a.borrow_mut().push(Active { memory, request, set, allocated, outside }));
}

pub fn exit() {
    ACTIVE.with(|a| a.borrow_mut().pop());
}

/// (live bytes, segments, shortfalls resolved, refusals, ns making segments).
pub fn stats() -> (u64, u64, u64, u64, u64) {
    (LIVE.load(Relaxed), SEGMENTS.load(Relaxed), RESOLVED.load(Relaxed), REFUSED.load(Relaxed), NS.load(Relaxed))
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

/// varena's answer to a shortfall of `need` bytes, for the thread's active pass.
fn ask(need: u64) -> Option<Next> {
    ACTIVE.with(|a| {
        let active = a.borrow();
        let p = active.last()?;
        let torch = Torch { allocated: p.allocated, reserved: p.outside + LIVE.load(Relaxed) };
        Some(p.request.oom(p.set, Some(need), None, torch))
    })
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
