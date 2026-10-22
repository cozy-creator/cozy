//! Torch pool segments, with recovery inside the failed allocation. All accounting is scoped
//! by CUDA device and pool. Callbacks never enter Python, manipulate the GIL, or wait for CUDA
//! events. Reclaim only touches completed Varena pages; other reclaim is done before execution.
//! A capture can consume prewarmed pool segments, but cannot grow the pool or enter recovery.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use tensorfs::io::cuda;
use varena::{Memory, Request, SetId, Torch, Workspace};

type Pool = (u64, u64);

#[derive(Clone)]
struct Active {
    memory: Arc<Memory>,
    request: Arc<Request>,
    set: Option<SetId>,
    outside: u64,
    device: i32,
    pool: Pool,
}

thread_local! {
    static ACTIVE: RefCell<Vec<Active>> = const { RefCell::new(Vec::new()) };
    static WORKSPACES: RefCell<Vec<Arc<Workspace>>> = const { RefCell::new(Vec::new()) };
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Default)]
struct Counters {
    live: u64,
    pending: u64,
    segments: u64,
    resolved: u64,
    resolved_bytes: u64,
    refused: u64,
    ns: u64,
    frees: u64,
}

#[derive(Clone, Copy)]
struct Allocation {
    device: i32,
    pool: Pool,
    bytes: u64,
    pending: bool,
}

#[derive(Default)]
struct Accounts {
    counts: BTreeMap<(i32, Pool), Counters>,
    pointers: BTreeMap<usize, Allocation>,
    faults: BTreeMap<i32, String>,
}

static ACCOUNTS: Mutex<Accounts> = Mutex::new(Accounts { counts: BTreeMap::new(), pointers: BTreeMap::new(), faults: BTreeMap::new() });

fn accounts() -> std::sync::MutexGuard<'static, Accounts> {
    ACCOUNTS.lock().unwrap_or_else(|p| p.into_inner())
}

fn live(device: i32) -> u64 {
    accounts().counts.iter().filter(|((d, _), _)| *d == device).map(|(_, c)| c.live).sum()
}

/// Segments torch has logically freed but whose GPU retirement has not completed.
pub fn pending(device: i32) -> u64 {
    accounts().counts.iter().filter(|((d, _), _)| *d == device).map(|(_, c)| c.pending).sum()
}

fn retiring(ptr: usize) -> bool {
    let mut all = accounts();
    let Some(a) = all.pointers.get_mut(&ptr) else { return false };
    if a.pending { return false; }
    a.pending = true;
    let a = *a;
    all.counts.entry((a.device, a.pool)).or_default().pending += a.bytes;
    true
}

pub fn fault_message(device: i32) -> Option<String> { accounts().faults.get(&device).cloned() }

fn quarantine(ptr: usize, reason: String) {
    let mut all = accounts();
    if let Some(a) = all.pointers.get(&ptr).copied() {
        all.faults.entry(a.device).or_insert(reason);
    }
}

pub fn inside() -> bool { INSIDE.with(Cell::get) }
// Retained for diagnostics of attempts to enter Python hooks from a native callback.
static TRIMS_INSIDE: AtomicU64 = AtomicU64::new(0);
static SHRINKS_INSIDE: AtomicU64 = AtomicU64::new(0);
pub fn trim_inside() { TRIMS_INSIDE.fetch_add(1, Relaxed); }
pub fn shrink_inside() { SHRINKS_INSIDE.fetch_add(1, Relaxed); }
pub fn reentries() -> (u64, u64) { (TRIMS_INSIDE.load(Relaxed), SHRINKS_INSIDE.load(Relaxed)) }

struct Inside(bool);
impl Inside {
    fn enter() -> Self { Self(INSIDE.with(|v| v.replace(true))) }
}
impl Drop for Inside {
    fn drop(&mut self) { INSIDE.with(|v| v.set(self.0)); }
}

pub fn enter(memory: Arc<Memory>, request: Arc<Request>, set: Option<SetId>, reserved: u64, device: i32, pool: Pool) {
    let outside = reserved.saturating_sub(live(device));
    ACTIVE.with(|a| a.borrow_mut().push(Active { memory, request, set, outside, device, pool }));
}
pub fn exit() { ACTIVE.with(|a| a.borrow_mut().pop()); }
pub fn workspace_enter(workspace: Arc<Workspace>) { WORKSPACES.with(|b| b.borrow_mut().push(workspace)); }
pub fn workspace_exit() { WORKSPACES.with(|b| b.borrow_mut().pop()); }

pub fn stats(device: Option<i32>, pool: Option<Pool>) -> (u64, u64, u64, u64, u64, u64, u64) {
    let c = accounts().counts.iter().filter(|((d, p), _)| device.is_none_or(|want| *d == want) && pool.is_none_or(|want| *p == want)).fold(Counters::default(), |mut a, (_, c)| {
        a.live += c.live; a.segments += c.segments; a.resolved += c.resolved;
        a.resolved_bytes += c.resolved_bytes; a.refused += c.refused; a.ns += c.ns; a.frees += c.frees; a
    });
    (c.live, c.segments, c.resolved, c.resolved_bytes, c.refused, c.ns, c.frees)
}

fn capturing(stream: *mut c_void) -> bool {
    let Ok(d) = cuda::driver() else { return true };
    let mut status = 0;
    // SAFETY: the caller provides torch's current CUDA stream; status is an out parameter.
    unsafe { (d.stream_is_capturing)(stream as cuda::CUstream, &mut status) != cuda::SUCCESS || status != 0 }
}

fn segment(size: usize) -> Result<cuda::CUdeviceptr, cuda::CUresult> {
    let d = cuda::driver().map_err(|_| -1)?;
    let mut ptr = 0;
    // SAFETY: an allocation in torch's current context with an out parameter on our stack.
    let rc = unsafe { (d.mem_alloc)(&mut ptr, size) };
    if rc == cuda::SUCCESS { Ok(ptr) } else { Err(rc) }
}

fn alloc(size: usize, device: i32, stream: *mut c_void) -> *mut c_void {
    let _inside = Inside::enter();
    if fault_message(device).is_some() { return std::ptr::null_mut(); }
    let began = Instant::now();
    let active = ACTIVE.with(|a| a.borrow().iter().rev().find(|p| p.device == device).cloned());
    let budget = active.as_ref().and_then(|a| WORKSPACES.with(|b| b.borrow().iter().rev().find(|b| b.belongs_to(&a.memory)).cloned()));
    let pool = active.as_ref().map_or((0, 0), |a| a.pool);
    let key = (device, pool);
    if capturing(stream) {
        accounts().counts.entry(key).or_default().refused += 1;
        return std::ptr::null_mut();
    }
    let mut previous = None;
    let mut reclaimed_total = 0;
    let ptr = loop {
        let held = active.as_ref().map_or(live(device), |a| a.outside.saturating_add(live(device)));
        let limit = match active.as_ref() {
            None => u64::MAX,
            Some(a) => match a.memory.allocator_limit() { Some(n) => n.saturating_add(budget.as_ref().map_or(0, |b| b.remaining())), None => break 0 },
        };
        if budget.as_ref().is_some_and(|b| b.remaining() < size as u64) { break 0; }
        let result = if held.saturating_add(size as u64) <= limit {
            let credit = match &budget {
                Some(b) => match b.take(size as u64) { Some(c) => Some(c), None => break 0 },
                None => None,
            };
            let result = segment(size);
            if result.is_ok() { if let Some(credit) = credit { credit.commit(); } }
            result
        } else { Err(cuda::ERROR_OUT_OF_MEMORY) };
        match result {
            Ok(ptr) => break ptr,
            Err(cuda::ERROR_OUT_OF_MEMORY) => {
                let Some(a) = active.as_ref() else { break 0 };
                // Neither a changed watermark nor a stale free count can spin forever. A new
                // attempt requires actual reclaimed bytes or a strictly larger available cap.
                let state = (held, limit);
                if previous == Some(state) { break 0; }
                previous = Some(state);
                let reclaimed = a.request.reclaim_allocation(a.set, size as u64, Torch { allocated: held, reserved: held }).unwrap_or(0);
                if reclaimed == 0 { break 0; }
                reclaimed_total += reclaimed;
            }
            Err(code) => {
                accounts().faults.entry(device).or_insert_with(|| format!("CUDA allocation failed with non-capacity error {code}"));
                break 0;
            }
        }
    };
    let mut all = accounts();
    let c = all.counts.entry(key).or_default();
    c.ns += began.elapsed().as_nanos() as u64;
    if ptr == 0 { c.refused += 1; }
    else {
        c.live += size as u64; c.segments += 1;
        if reclaimed_total != 0 { c.resolved += 1; c.resolved_bytes += reclaimed_total; }
        all.pointers.insert(ptr as usize, Allocation { device, pool, bytes: size as u64, pending: false });
    }
    ptr as *mut c_void
}

#[no_mangle]
pub extern "C" fn varena_torch_alloc(size: isize, device: c_int, stream: *mut c_void) -> *mut c_void {
    let Ok(size) = usize::try_from(size) else { return std::ptr::null_mut() };
    catch_unwind(AssertUnwindSafe(|| alloc(size, device, stream))).unwrap_or(std::ptr::null_mut())
}

struct Pending { ptr: usize, event: usize, context: usize }
static FREER: OnceLock<mpsc::Sender<Pending>> = OnceLock::new();

fn freer() -> &'static mpsc::Sender<Pending> {
    FREER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Pending>();
        // The worker owns waits; a CUDA host callback is free to acquire the GIL or torch's
        // lock while this thread waits. An allocating thread never waits for this worker.
        std::thread::spawn(move || {
            while let Ok(p) = rx.recv() {
                let Ok(d) = cuda::driver() else { quarantine(p.ptr, "CUDA driver unavailable during retirement".into()); continue };
                let Ok(_ctx) = cuda::CtxGuard::enter(p.context as cuda::CUcontext) else { quarantine(p.ptr, "CUDA context unavailable during retirement".into()); continue };
                // SAFETY: this worker owns the event and the segment is no longer torch's.
                let ready = unsafe { (d.event_synchronize)(p.event as cuda::CUevent) };
                if ready != cuda::SUCCESS { quarantine(p.ptr, format!("consumer completion could not be proven (CUDA {ready})")); continue; }
                let rc = unsafe { (d.mem_free)(p.ptr as cuda::CUdeviceptr) };
                unsafe { (d.event_destroy)(p.event as cuda::CUevent); }
                if rc == cuda::SUCCESS {
                    let mut all = accounts();
                    if let Some(a) = all.pointers.remove(&p.ptr) {
                        let c = all.counts.entry((a.device, a.pool)).or_default();
                        c.live = c.live.saturating_sub(a.bytes);
                        if a.pending { c.pending = c.pending.saturating_sub(a.bytes); }
                        c.frees += 1;
                    }
                } else { quarantine(p.ptr, format!("CUDA segment release failed ({rc})")); }
            }
        });
        tx
    })
}

#[no_mangle]
pub extern "C" fn varena_torch_free(ptr: *mut c_void, _size: usize, _device: c_int, stream: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _inside = Inside::enter();
        // Graph-owned pools must only be freed after graph disposal; never synchronize or
        // record a real completion event into an in-progress capture.
        if ptr.is_null() || !retiring(ptr as usize) { return; }
        if capturing(stream) { quarantine(ptr as usize, "segment freed during capture or capture state unknown".into()); return; }
        let Ok(d) = cuda::driver() else { quarantine(ptr as usize, "CUDA driver unavailable during free".into()); return };
        let (mut context, mut event) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: context/stream belong to torch; the newly-created event is ours.
        unsafe {
            if (d.ctx_get_current)(&mut context) != cuda::SUCCESS || (d.event_create)(&mut event, cuda::EVENT_DEFAULT) != cuda::SUCCESS {
                quarantine(ptr as usize, "unable to create a retirement event".into()); return;
            }
            if (d.event_record)(event, stream as cuda::CUstream) != cuda::SUCCESS {
                (d.event_destroy)(event);
                quarantine(ptr as usize, "unable to record segment completion".into()); return;
            }
        }
        if freer().send(Pending { ptr: ptr as usize, event: event as usize, context: context as usize }).is_err() {
            quarantine(ptr as usize, "native retirement worker stopped".into());
        }
    }));
}
