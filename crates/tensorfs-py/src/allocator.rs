//! Torch pool segments, with recovery inside the failed allocation. All accounting is scoped
//! by CUDA device and pool. Callbacks never enter Python, manipulate the GIL, or wait for CUDA
//! events. Reclaim only touches completed Varena pages; other reclaim is done before execution.
//! A capture can consume prewarmed pool segments, but cannot grow the pool or enter recovery.
//! Driver calls run in relaxed capture mode, as torch's own capture-time cudaMalloc does: they
//! never invalidate another thread's capture. A segment whose release cannot be ordered (freed
//! during capture, or its completion unrecordable) is quarantined: it stays charged and is never
//! reused. Only a sticky CUDA error or unprovable ownership faults the device.

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
pub type Refusal = (i32, Pool, &'static str, u64, u64, u64);

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

/// CU_STREAM_CAPTURE_MODE_RELAXED.
const RELAXED: c_int = 2;

/// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED ..= CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD.
fn capture_error(code: cuda::CUresult) -> bool {
    (900..=908).contains(&code)
}

/// The context is gone (varena's classification of sticky CUDA errors).
fn sticky(code: cuda::CUresult) -> bool {
    matches!(code, 700 | 702 | 709 | 710 | 714..=719 | 721 | 911)
}

/// This thread in relaxed capture mode until dropped (its previous mode then comes back).
struct RelaxedCapture(Option<c_int>);

impl RelaxedCapture {
    fn enter() -> Self {
        let Ok(d) = cuda::driver() else { return Self(None) };
        let mut mode = RELAXED;
        // SAFETY: the driver ABI; `mode` is an in/out parameter on our stack.
        Self((unsafe { (d.thread_exchange_stream_capture_mode)(&mut mode) } == cuda::SUCCESS).then_some(mode))
    }
}

impl Drop for RelaxedCapture {
    fn drop(&mut self) {
        if let (Some(mut mode), Ok(d)) = (self.0, cuda::driver()) {
            // SAFETY: as above; the mode this thread had before.
            unsafe { (d.thread_exchange_stream_capture_mode)(&mut mode) };
        }
    }
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
    quarantined: u64,
    refusal: Option<(&'static str, u64, u64, u64)>,
    quarantine: Option<&'static str>,
}

struct Allocation {
    device: i32,
    context: usize,
    pool: Pool,
    bytes: u64,
    pending: bool,
    _resident: Option<varena::memory::Resident>,
}

#[derive(Default)]
struct Accounts {
    counts: BTreeMap<(i32, Pool), Counters>,
    pointers: BTreeMap<usize, Allocation>,
    faults: BTreeMap<i32, String>,
    quarantined: Vec<Allocation>,
}

impl Accounts {
    fn detach(&mut self, ptr: usize) -> Option<Allocation> { self.pointers.remove(&ptr) }

    fn released(&mut self, allocation: &Allocation) {
        let c = self.counts.entry((allocation.device, allocation.pool)).or_default();
        c.live = c.live.saturating_sub(allocation.bytes);
        if allocation.pending { c.pending = c.pending.saturating_sub(allocation.bytes); }
        c.frees += 1;
    }

    fn failed_release(&mut self, allocation: Allocation, reason: String) {
        self.faults.entry(allocation.device).or_insert(reason);
        self.quarantined.push(allocation);
    }
}

static ACCOUNTS: Mutex<Accounts> = Mutex::new(Accounts { counts: BTreeMap::new(), pointers: BTreeMap::new(), faults: BTreeMap::new(), quarantined: Vec::new() });

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

fn retiring(ptr: usize) -> Option<(i32, usize)> {
    let mut all = accounts();
    let a = all.pointers.get_mut(&ptr)?;
    if a.pending { return None; }
    a.pending = true;
    let (device, pool, bytes, context) = (a.device, a.pool, a.bytes, a.context);
    all.counts.entry((device, pool)).or_default().pending += bytes;
    Some((device, context))
}

pub fn fault_message(device: i32) -> Option<String> { accounts().faults.get(&device).cloned() }

/// Ownership of the segment at `ptr` cannot be proved: the device's executor retires.
fn fault(ptr: usize, reason: String) {
    let mut all = accounts();
    if let Some(device) = all.pointers.get(&ptr).map(|a| a.device) {
        all.faults.entry(device).or_insert(reason);
    }
}

/// The segment at `ptr` stays charged (pending) and is never freed or reused; the device goes on.
fn quarantine(ptr: usize, reason: &'static str) {
    let mut all = accounts();
    let Some((device, pool, bytes)) = all.pointers.get(&ptr).map(|a| (a.device, a.pool, a.bytes)) else { return };
    let c = all.counts.entry((device, pool)).or_default();
    c.quarantined += bytes;
    c.quarantine = Some(reason);
}

/// Per device and pool: bytes quarantined and the last reason.
pub fn quarantined(device: Option<i32>) -> Vec<(i32, Pool, u64, &'static str)> {
    accounts().counts.iter().filter(|((d, _), _)| device.is_none_or(|want| *d == want))
        .filter_map(|((d, p), c)| c.quarantine.map(|why| (*d, *p, c.quarantined, why))).collect()
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

/// Last refusal by device/pool. Collected outside the callback: no callback logging or Python.
pub fn refusals(device: Option<i32>) -> Vec<Refusal> {
    accounts().counts.iter().filter(|((d, _), _)| device.is_none_or(|want| *d == want))
        .filter_map(|((d, p), c)| c.refusal.map(|(why, size, held, limit)| (*d, *p, why, size, held, limit))).collect()
}

fn capturing(stream: *mut c_void) -> bool {
    let Ok(d) = cuda::driver() else { return true };
    let mut status = 0;
    // SAFETY: the caller provides torch's current CUDA stream; status is an out parameter.
    unsafe { (d.stream_is_capturing)(stream as cuda::CUstream, &mut status) != cuda::SUCCESS || status != 0 }
}

fn segment(size: usize) -> Result<cuda::CUdeviceptr, cuda::CUresult> {
    let d = cuda::driver().map_err(|_| -1)?;
    let _relaxed = RelaxedCapture::enter();
    let mut ptr = 0;
    // SAFETY: an allocation in torch's current context with an out parameter on our stack.
    let rc = unsafe { (d.mem_alloc)(&mut ptr, size) };
    if rc == cuda::SUCCESS { Ok(ptr) } else { Err(rc) }
}

fn alloc(size: usize, device: i32, stream: *mut c_void) -> *mut c_void {
    let _inside = Inside::enter();
    if fault_message(device).is_some() {
        let held = live(device);
        let mut all = accounts();
        let pool = ACTIVE.with(|a| a.borrow().iter().rev().find(|p| p.device == device).map_or((0, 0), |a| a.pool));
        let c = all.counts.entry((device, pool)).or_default();
        c.refused += 1;
        c.refusal = Some(("ownership_fault", size as u64, held, 0));
        return std::ptr::null_mut();
    }
    let Ok(driver) = cuda::driver() else { return std::ptr::null_mut() };
    let mut context = std::ptr::null_mut();
    // Torch's allocation context belongs to the pointer, not to its eventual freeing thread.
    if unsafe { (driver.ctx_get_current)(&mut context) } != cuda::SUCCESS || context.is_null() {
        accounts().faults.entry(device).or_insert_with(|| "allocation has no current CUDA context".into());
        return std::ptr::null_mut();
    }
    let began = Instant::now();
    let active = ACTIVE.with(|a| a.borrow().iter().rev().find(|p| p.device == device).cloned());
    let budget = active.as_ref().and_then(|a| WORKSPACES.with(|b| b.borrow().iter().rev().find(|b| b.belongs_to(&a.memory)).cloned()));
    let pool = active.as_ref().map_or((0, 0), |a| a.pool);
    let key = (device, pool);
    if capturing(stream) {
        let held = live(device);
        let mut all = accounts();
        let c = all.counts.entry(key).or_default();
        c.refused += 1;
        c.refusal = Some(("capture_growth", size as u64, held, 0));
        return std::ptr::null_mut();
    }
    let mut previous = None;
    let mut reclaimed_total = 0;
    let mut refusal;
    let ptr = loop {
        let held = active.as_ref().map_or(live(device), |a| a.outside.saturating_add(live(device)));
        let limit = match active.as_ref() {
            None => u64::MAX,
            Some(a) => match a.memory.allocator_limit() { Some(n) => n.saturating_add(budget.as_ref().map_or(0, |b| b.remaining())), None => { refusal = ("state_unavailable", size as u64, held, 0); break 0 } },
        };
        refusal = ("workspace", size as u64, held, limit);
        if budget.as_ref().is_some_and(|b| b.remaining() < size as u64) { break 0; }
        let result = if held.saturating_add(size as u64) <= limit {
            let credit = match &budget {
                Some(b) => match b.take(size as u64) { Some(c) => Some(c), None => { refusal = ("state_unavailable", size as u64, held, 0); break 0 } },
                None => None,
            };
            let result = segment(size);
            if result.is_ok() { if let Some(credit) = credit { credit.commit(); } }
            result
        } else { Err(cuda::ERROR_OUT_OF_MEMORY) };
        match result {
            Ok(ptr) => break ptr,
            Err(cuda::ERROR_OUT_OF_MEMORY) => {
                refusal = ("driver_capacity", size as u64, held, limit);
                let Some(a) = active.as_ref() else { break 0 };
                // Neither a changed watermark nor a stale free count can spin forever. A new
                // attempt requires actual reclaimed bytes or a strictly larger available cap.
                let state = (held, limit);
                refusal.0 = "no_progress";
                if previous == Some(state) { break 0; }
                previous = Some(state);
                let reclaimed = a.request.reclaim_allocation(a.set, size as u64, Torch { allocated: held, reserved: held }).unwrap_or(0);
                refusal.0 = "no_reclaim";
                if reclaimed == 0 { break 0; }
                reclaimed_total += reclaimed;
            }
            Err(code) if capture_error(code) => {
                refusal = ("capture", size as u64, held, limit);
                break 0;
            }
            Err(code) => {
                refusal = ("driver_error", size as u64, held, limit);
                if sticky(code) {
                    accounts().faults.entry(device).or_insert_with(|| format!("CUDA allocation failed with sticky error {code}"));
                }
                break 0;
            }
        }
    };
    let resident = (ptr != 0).then(|| active.as_ref().map(|a| a.memory.resident(size as u64))).flatten();
    let mut all = accounts();
    let c = all.counts.entry(key).or_default();
    c.ns += began.elapsed().as_nanos() as u64;
    if ptr == 0 { c.refused += 1; c.refusal = Some(refusal); }
    else {
        c.live += size as u64; c.segments += 1;
        if reclaimed_total != 0 { c.resolved += 1; c.resolved_bytes += reclaimed_total; }
        all.pointers.insert(ptr as usize, Allocation { device, context: context as usize, pool, bytes: size as u64, pending: false, _resident: resident });
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
            // Its frees never invalidate another thread's capture.
            let _relaxed = RelaxedCapture::enter();
            while let Ok(p) = rx.recv() {
                let Ok(d) = cuda::driver() else { quarantine(p.ptr, "retirement without a CUDA driver"); continue };
                let Ok(_ctx) = cuda::CtxGuard::enter(p.context as cuda::CUcontext) else { quarantine(p.ptr, "retirement without its CUDA context"); continue };
                // SAFETY: this worker owns the event and the segment is no longer torch's.
                let ready = unsafe { (d.event_synchronize)(p.event as cuda::CUevent) };
                if ready != cuda::SUCCESS {
                    if sticky(ready) { fault(p.ptr, format!("consumer completion could not be proven (CUDA {ready})")); }
                    quarantine(p.ptr, "consumer completion unproven");
                    continue;
                }
                // Detach this generation before the driver can reuse its address. A new
                // allocation at the same pointer must never be removed by this old free.
                let Some(a) = accounts().detach(p.ptr) else { continue };
                let rc = unsafe { (d.mem_free)(p.ptr as cuda::CUdeviceptr) };
                unsafe { (d.event_destroy)(p.event as cuda::CUevent); }
                if rc == cuda::SUCCESS {
                    {
                        accounts().released(&a);
                    }
                    drop(a); // physical token/public owner drops outside the accounts lock
                } else {
                    accounts().failed_release(a, format!("CUDA segment release failed ({rc})"));
                }
            }
        });
        tx
    })
}

#[no_mangle]
pub extern "C" fn varena_torch_free(ptr: *mut c_void, _size: usize, device: c_int, stream: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _inside = Inside::enter();
        // Graph-owned pools must only be freed after graph disposal; never synchronize or
        // record a real completion event into an in-progress capture.
        if ptr.is_null() { return; }
        let Some((owner_device, context)) = retiring(ptr as usize) else { return };
        if device != owner_device { fault(ptr as usize, "free device differs from allocation owner".into()); return; }
        let Ok(_ctx) = cuda::CtxGuard::enter(context as cuda::CUcontext) else { quarantine(ptr as usize, "freed without its CUDA context"); return };
        // An event recorded into a capture would be a graph node, not this segment's completion.
        if capturing(stream) { quarantine(ptr as usize, "freed during capture"); return; }
        let Ok(d) = cuda::driver() else { quarantine(ptr as usize, "freed without a CUDA driver"); return };
        let _relaxed = RelaxedCapture::enter();
        let mut event = std::ptr::null_mut();
        // SAFETY: context/stream belong to torch; the newly-created event is ours.
        unsafe {
            let rc = (d.event_create)(&mut event, cuda::EVENT_DEFAULT);
            if rc != cuda::SUCCESS {
                if sticky(rc) { fault(ptr as usize, format!("retirement event creation failed (CUDA {rc})")); }
                quarantine(ptr as usize, "no retirement event"); return;
            }
            let rc = (d.event_record)(event, stream as cuda::CUstream);
            if rc != cuda::SUCCESS {
                (d.event_destroy)(event);
                if sticky(rc) { fault(ptr as usize, format!("segment completion unrecordable (CUDA {rc})")); }
                quarantine(ptr as usize, "completion unrecordable"); return;
            }
        }
        if freer().send(Pending { ptr: ptr as usize, event: event as usize, context }).is_err() {
            quarantine(ptr as usize, "retirement worker stopped");
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocation(pool: Pool, bytes: u64, pending: bool) -> Allocation {
        Allocation { device: 0, context: 1, pool, bytes, pending, _resident: None }
    }

    #[test]
    fn an_old_free_cannot_remove_a_new_allocation_at_the_reused_address() {
        let mut a = Accounts::default();
        let ptr = 0x1000;
        a.pointers.insert(ptr, allocation((1, 0), 16, true));
        a.counts.insert((0, (1, 0)), Counters { live: 16, pending: 16, ..Counters::default() });
        let old = a.detach(ptr).unwrap(); // immediately before cuMemFree
        // The driver returns the same address before the old worker updates its counters.
        a.pointers.insert(ptr, allocation((2, 0), 32, false));
        a.counts.insert((0, (2, 0)), Counters { live: 32, ..Counters::default() });
        a.released(&old);
        assert_eq!(a.pointers[&ptr].pool, (2, 0));
        assert_eq!((a.counts[&(0, (1, 0))].live, a.counts[&(0, (1, 0))].pending), (0, 0));
        assert_eq!(a.counts[&(0, (2, 0))].live, 32);
    }

    #[test]
    fn an_uncertain_free_keeps_its_detached_allocation_charged() {
        let mut a = Accounts::default();
        a.pointers.insert(0x1000, allocation((1, 0), 16, true));
        a.counts.insert((0, (1, 0)), Counters { live: 16, pending: 16, ..Counters::default() });
        let old = a.detach(0x1000).unwrap();
        a.failed_release(old, "release unproved".into());
        assert_eq!((a.counts[&(0, (1, 0))].live, a.counts[&(0, (1, 0))].pending), (16, 16));
        assert_eq!(a.quarantined[0].bytes, 16);
        assert!(a.faults.contains_key(&0));
    }
}
