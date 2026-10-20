//! Private real-CUDA controls. Every selected case runs in its own bounded process.
//! Ordinary CPU tests never load CUDA; explicit GPU selection refuses missing opt-in/capability.

use super::*;
use crate::cuda::proof;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);
const WATCHDOG: Duration = Duration::from_secs(20);
const BYTES: u64 = REGION_ALIGN;
const CASES: &[&str] = &[
    "source_handoff",
    "consumer_dependency",
    "home_record",
    "streamed_record",
    "staging_prior",
    "last_pin",
    "healthy_drain",
    "unregister",
    "secondary_consumer_ready_positive",
    "secondary_consumer_partial_record_negative",
];

#[derive(Default)]
struct GateFlags {
    entered: bool,
    released: bool,
    exited: bool,
    expired: bool,
}
#[derive(Default)]
struct GateState {
    flags: Mutex<GateFlags>,
    cv: Condvar,
}

impl GateState {
    fn release(&self) {
        self.flags
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .released = true;
        self.cv.notify_all();
    }

    fn wait(&self, what: &str, ready: impl Fn(&GateFlags) -> bool) {
        let state = self.flags.lock().unwrap_or_else(|e| e.into_inner());
        let (state, _) = self
            .cv
            .wait_timeout_while(state, WAIT, |s| !ready(s))
            .unwrap_or_else(|e| e.into_inner());
        assert!(
            ready(&state) && !state.expired,
            "REFUSED: gate {what} was not acknowledged"
        );
    }
}

// No CUDA, user callback, panic, logging, or dependency on later GPU work in this function.
unsafe extern "C" fn host_gate(data: *mut c_void) {
    // SAFETY: arm passes one retained Arc raw reference; only this callback consumes it.
    let gate = unsafe { Arc::<GateState>::from_raw(data.cast()) };
    let mut state = gate.flags.lock().unwrap_or_else(|e| e.into_inner());
    state.entered = true;
    gate.cv.notify_all();
    while !state.released {
        state = gate.cv.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    state.exited = true;
    gate.cv.notify_all();
}

struct Gate {
    state: Arc<GateState>,
    done: Option<mpsc::Sender<()>>,
    watchdog: Option<JoinHandle<()>>,
}

impl Gate {
    fn new() -> Self {
        let state = Arc::new(GateState::default());
        let (done, rx) = mpsc::channel();
        let guard = state.clone();
        let watchdog = std::thread::spawn(move || {
            if matches!(
                rx.recv_timeout(WATCHDOG),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                let mut flags = guard.flags.lock().unwrap_or_else(|e| e.into_inner());
                if !flags.released {
                    flags.expired = true;
                    flags.released = true;
                    guard.cv.notify_all();
                }
            }
        });
        Self {
            state,
            done: Some(done),
            watchdog: Some(watchdog),
        }
    }

    fn arm(stream: CUstream) -> Self {
        let api = proof::symbols().expect("REFUSED: optional CUDA proof symbols unavailable");
        let gate = Self::new();
        let data = Arc::into_raw(gate.state.clone()) as *mut c_void;
        // SAFETY: a live precreated stream; raw userdata remains retained even on ambiguous error.
        let result = unsafe { (api.host)(stream, host_gate, data) };
        cuda::check("proof cuLaunchHostFunc", result)
            .expect("REFUSED: CUDA host gate not accepted");
        gate.state.wait("entered", |s| s.entered);
        println!("VARENA_GPU_PHASE {{\"phase\":\"gate_entered\"}}");
        gate
    }

    fn release(&self) {
        self.state.release();
        self.state.wait("exited", |s| s.exited);
        println!("VARENA_GPU_PHASE {{\"phase\":\"gate_exited\"}}");
    }

    fn check(&self) {
        assert!(
            !self
                .state
                .flags
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .expired,
            "REFUSED: watchdog released the gate; not a proof"
        );
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.state.release();
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

#[test]
fn cpu_gate_release_acknowledges_callback_without_cuda() {
    let gate = Gate::new();
    let raw = Arc::into_raw(gate.state.clone()) as usize;
    let callback = std::thread::spawn(move || {
        // SAFETY: this CPU control transfers exactly one Arc raw reference to the callback.
        unsafe { host_gate(raw as *mut c_void) };
    });
    gate.state.wait("entered", |s| s.entered);
    gate.release();
    callback.join().unwrap();
    gate.check();
}

#[test]
fn cpu_gate_drop_releases_callback_without_cuda() {
    let gate = Gate::new();
    let state = gate.state.clone();
    let raw = Arc::into_raw(state.clone()) as usize;
    let callback = std::thread::spawn(move || {
        // SAFETY: one retained raw Arc, consumed once by this callback.
        unsafe { host_gate(raw as *mut c_void) };
    });
    state.wait("entered", |s| s.entered);
    drop(gate);
    state.wait("exited", |s| s.exited);
    callback.join().unwrap();
}

#[test]
#[ignore = "requires explicit opt-in, exact source and one real CUDA device; one process per case"]
fn gated_cuda_custody() {
    assert_eq!(
        std::env::var("VARENA_GPU_PROOF").as_deref(),
        Ok("1"),
        "REFUSED: explicit opt-in missing"
    );
    let source = option_env!("VARENA_GPU_PROOF_SOURCE").unwrap_or("");
    assert!(
        source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()),
        "REFUSED: embedded source absent"
    );
    assert_eq!(
        std::env::var("VARENA_GPU_PROOF_SOURCE").as_deref(),
        Ok(source),
        "REFUSED: source differs"
    );
    let case = std::env::var("VARENA_GPU_CASE").expect("REFUSED: case missing");
    assert!(CASES.contains(&case.as_str()), "REFUSED: unknown case");
    let d = cuda::driver().expect("REFUSED: CUDA driver unavailable");
    let mut count = 0;
    // SAFETY: initialized driver and a live output parameter.
    checked("cuDeviceGetCount", unsafe {
        (d.device_get_count)(&mut count)
    });
    assert_eq!(
        count, 1,
        "REFUSED: proof requires exactly the one assigned GPU"
    );
    proof::symbols().expect("REFUSED: optional proof symbols absent");
    match case.as_str() {
        "source_handoff" => source_handoff(),
        "consumer_dependency" => consumer_dependency(),
        "home_record" => record_failure(true),
        "streamed_record" => record_failure(false),
        "staging_prior" => staging_prior(),
        "last_pin" => last_pin(),
        "healthy_drain" => healthy_drain(),
        "unregister" => unregister(),
        "secondary_consumer_ready_positive" => secondary_consumer(false),
        "secondary_consumer_partial_record_negative" => secondary_consumer(true),
        _ => unreachable!(),
    }
    let mut dev = 0;
    let mut uuid = [0u8; 16];
    // SAFETY: initialized driver, selected ordinal and correctly sized output values.
    checked("cuDeviceGet", unsafe { (d.device_get)(&mut dev, 0) });
    // SAFETY: the UUID buffer is the documented16-byte output.
    checked("cuDeviceGetUuid", unsafe {
        (d.device_get_uuid)(&mut uuid, dev)
    });
    let uuid: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        r#"VARENA_GPU_CUSTODY {{"schema":1,"status":"pass","case":"{case}","source":"{source}","device_count":1,"device_uuid":"{uuid}","gate":"cuLaunchHostFunc","consumer":"cuMemcpyDtoHAsync_v2","actual_cuda":true}}"#
    );
}

fn checked(op: &'static str, rc: i32) {
    cuda::check(op, rc).unwrap_or_else(|e| panic!("REFUSED or failed CUDA proof operation: {e}"));
}

fn phase(rx: mpsc::Receiver<()>, what: &str) {
    rx.recv_timeout(WAIT)
        .unwrap_or_else(|e| panic!("REFUSED: actual {what} phase absent: {e}"));
    println!("VARENA_GPU_PHASE {{\"phase\":{what:?}}}");
}

fn sync(gpu: &Gpu, stream: CUstream) {
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    // SAFETY: a live stream created in this fixture's current primary context.
    checked("proof stream completion", unsafe {
        (cuda::driver().unwrap().stream_synchronize)(stream)
    });
}

struct Consumer {
    stream: u64,
    ctx: usize,
    ptr: usize,
    completed: AtomicBool,
}
impl Consumer {
    fn new(gpu: &Gpu) -> Self {
        let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
        let d = cuda::driver().unwrap();
        let mut stream = std::ptr::null_mut();
        let mut ptr = std::ptr::null_mut();
        // SAFETY: valid output handles; allocations occur before any gate.
        checked("proof consumer stream", unsafe {
            (d.stream_create)(&mut stream, cuda::STREAM_NON_BLOCKING)
        });
        // SAFETY: valid output pointer, bounded pinned readback allocation.
        checked("proof readback allocation", unsafe {
            (d.mem_host_alloc)(&mut ptr, BYTES as usize, cuda::MEMHOSTALLOC_PORTABLE)
        });
        Self {
            stream: stream as u64,
            ctx: gpu.inner.ctx as usize,
            ptr: ptr as usize,
            completed: AtomicBool::new(true),
        }
    }

    fn enqueue(&self, va: u64) {
        self.completed.store(false, Ordering::Release);
        let _c = CtxGuard::enter(self.ctx as CUcontext).unwrap();
        let api = proof::symbols().unwrap();
        // SAFETY: actual live GPU mapping and our preallocated pinned destination, both BYTES long.
        checked("proof consumer D2H enqueue", unsafe {
            (api.dtoh)(
                self.ptr as *mut c_void,
                va,
                BYTES as usize,
                self.stream as CUstream,
            )
        });
        println!("VARENA_GPU_PHASE {{\"phase\":\"consumer_enqueued\"}}");
    }

    fn verify(&self, gpu: &Gpu, byte: u8) {
        sync(gpu, self.stream as CUstream);
        self.completed.store(true, Ordering::Release);
        // SAFETY: D2H completed into our live pinned buffer.
        let bytes = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, BYTES as usize) };
        assert!(
            bytes.iter().all(|&b| b == byte),
            "real GPU readback differs"
        );
        println!("VARENA_GPU_WITNESS {{\"kind\":\"readback\",\"bytes\":{BYTES},\"byte\":{byte}}}");
    }
}
impl Drop for Consumer {
    fn drop(&mut self) {
        // A failed/ambiguous enqueue or unwinding assertion cannot authorize freeing a D2H
        // destination. These bounded private child resources stay live until process exit.
        if !self.completed.load(Ordering::Acquire) || std::thread::panicking() {
            return;
        }
        if let (Ok(_c), Ok(d)) = (CtxGuard::enter(self.ctx as CUcontext), cuda::driver()) {
            // SAFETY: fixture-owned handles; every gate's release guard drops before this object.
            unsafe {
                (d.stream_destroy)(self.stream as CUstream);
                (d.mem_free_host)(self.ptr as *mut c_void);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Fd {
    raw: i32,
    dev: u64,
    ino: u64,
}
impl Fd {
    fn new(file: &std::fs::File) -> Self {
        let m = file.metadata().unwrap();
        Self {
            raw: file.as_raw_fd(),
            dev: m.dev(),
            ino: m.ino(),
        }
    }
    fn alive(self) -> bool {
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: fstat only inspects our descriptor, writing a correctly sized stat on success.
        if unsafe { libc::fstat(self.raw, st.as_mut_ptr()) } != 0 {
            return false;
        }
        // SAFETY: successful fstat initialized the value.
        let st = unsafe { st.assume_init() };
        st.st_dev == self.dev && st.st_ino == self.ino
    }
}

struct SetFixture {
    ws: Option<Arc<WeightSet>>,
    source: Weak<Weights>,
    fd: Fd,
    byte: u8,
}
impl SetFixture {
    fn new(gpu: &Arc<Gpu>, host: bool, home: bool, byte: u8) -> Self {
        let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
        let data = vec![byte; BYTES as usize];
        // SAFETY: constant NUL-terminated name; a new fixture-owned anonymous file.
        let raw = unsafe { libc::memfd_create(c"varena-gpu-proof".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(raw >= 0, "REFUSED: source memfd unavailable");
        // SAFETY: the successful memfd call transferred a new descriptor.
        let mut file = unsafe { std::fs::File::from_raw_fd(raw) };
        file.write_all(&data).unwrap();
        let fd = Fd::new(&file);
        let object = tensorfs::ids::ObjectRef::of(&data);
        let layout = Layout {
            parts: vec![],
            nbytes: BYTES,
            digest: object.sha256.clone(),
            names: vec![vec![]],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: BYTES,
                span: BYTES,
                items: vec![Item {
                    offset: 0,
                    len: BYTES,
                    source: ItemSource::Object(tensorfs::read::ObjectRange {
                        obj: object.clone(),
                        off: 0,
                        len: BYTES,
                    }),
                    part_end: true,
                }],
            }],
        };
        let host = host.then(|| {
            let h = HostMem::create("gpu-proof-source", &layout).unwrap();
            // SAFETY: our new writable host layout has BYTES of data and no readers yet.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), h.base(), data.len()) };
            h.set_ready(0, true);
            // Preserve the ordinary first-copy policy, then establish actual registration
            // before the gate. No registration attempt may be deferred into gated work.
            assert!(!h.pin_region(0, 0, BYTES), "first-copy policy changed");
            assert!(
                h.pin_region(0, 0, BYTES),
                "REFUSED: direct source registration unsupported"
            );
            h
        });
        let weights = Arc::new(Weights {
            name: "gpu-proof".into(),
            parts: vec![],
            layout,
            source: Source::files(vec![(object, file)], false, 0),
            host,
        });
        let weak = Arc::downgrade(&weights);
        let g = &gpu.inner;
        let mut st = g.lock();
        let base = reserve_va(BYTES).unwrap();
        let map = home.then(|| g.map(&mut st, base, BYTES, false).unwrap());
        let z = Counters::default();
        let id = st.sets.len();
        st.sets.push(Some(Set {
            w: weights.clone(),
            pending: vec![],
            base,
            regs: vec![Reg { home, map }],
            step: z,
            last: z,
            total: z,
            used: 0,
            warm: true,
            busy: false,
            shared: false,
        }));
        if !home {
            st.slot_span = BYTES;
            if st.slots.is_empty() {
                g.new_slot(&mut st, BYTES).unwrap();
            }
        }
        drop(st);
        Self {
            ws: Some(Arc::new(WeightSet {
                gpu: gpu.clone(),
                id,
                base,
                weights,
            })),
            source: weak,
            fd,
            byte,
        }
    }

    fn ws(&self) -> &Arc<WeightSet> {
        self.ws.as_ref().unwrap()
    }
    fn assert_source(&self) {
        assert!(
            self.source.upgrade().is_some() && self.fd.alive(),
            "actual source owner/FD lost"
        );
        println!(
            "VARENA_GPU_WITNESS {{\"kind\":\"source\",\"fd\":{},\"inode\":{}}}",
            self.fd.raw, self.fd.ino
        );
        if let Some(w) = self.source.upgrade() {
            if let Some(h) = &w.host {
                let mut resident = 0u8;
                // SAFETY: mincore inspects one page of the retained actual host mapping.
                assert_eq!(
                    unsafe { libc::mincore(h.base().cast(), PART_ALIGN as usize, &mut resident) },
                    0
                );
            }
        }
    }
}

fn setup() -> (Arc<Gpu>, Consumer) {
    let gpu = Gpu::open(0, REGION_ALIGN).expect("REFUSED: real VMM GPU unavailable");
    let consumer = Consumer::new(&gpu);
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let mut events = vec![];
    // Precreate event resources before any callback can block a stream.
    for timing in [false, true] {
        for _ in 0..64 {
            events.push(gpu.inner.events.record(gpu.inner.copy, timing).unwrap());
        }
    }
    sync(&gpu, gpu.inner.copy);
    drop(events);
    (gpu, consumer)
}

fn queue(set: &SetFixture, home: bool) -> (At, Arc<Fill>, u64, Vec<Handle>) {
    let ws = set.ws();
    let g = &ws.gpu.inner;
    let _c = CtxGuard::enter(g.ctx).unwrap();
    let mut st = g.lock();
    let at = if home {
        let m = st.reg(ws.id, 0).unwrap().map.as_mut().unwrap();
        let ((fill, waits), dst) = (m.refill(), m.va);
        let w = st.set(ws.id).w.clone();
        st.set(ws.id).track_fill(fill.clone());
        g.submit(
            Job {
                set: ws.id,
                w,
                region: 0,
                dst,
                waits,
                fill,
            },
            true,
        );
        At::Home
    } else {
        At::Slot(g.slot_for(&mut st, ws.id, 0, true).unwrap().unwrap())
    };
    let m = match at {
        At::Home => st.reg(ws.id, 0).unwrap().map.as_ref().unwrap(),
        At::Slot(i) => &st.slot(i).m,
    };
    (
        at,
        m.fill.as_ref().unwrap().clone(),
        m.va,
        m.chunks.iter().map(|c| c.handle).collect(),
    )
}

fn same_mapping(gpu: &Gpu, id: usize, at: At, va: u64, handles: &[Handle]) {
    let mut st = gpu.inner.lock();
    let m = match at {
        At::Home => st.reg(id, 0).unwrap().map.as_ref().unwrap(),
        At::Slot(i) => &st.slot(i).m,
    };
    assert_eq!(m.va, va);
    assert_eq!(
        m.chunks.iter().map(|c| c.handle).collect::<Vec<_>>(),
        handles
    );
    println!("VARENA_GPU_WITNESS {{\"kind\":\"mapping\",\"va\":{va},\"handles\":{handles:?}}}");
}

fn closed(gpu: &Gpu, gate: &Gate) {
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let marker = gpu.inner.events.record(gpu.inner.copy, false).unwrap();
    assert!(
        !marker.done().unwrap(),
        "gate did not hold actual queued producer work"
    );
    gate.check();
}

fn warm(gpu: &Gpu, set: &SetFixture, home: bool) -> u64 {
    let (_, fill, va, _) = queue(set, home);
    fill.wait().unwrap();
    sync(gpu, gpu.inner.copy);
    va
}

fn record_failure(home: bool) {
    let (gpu, consumer) = setup();
    let set = SetFixture::new(&gpu, home, home, 0x5a);
    let id = set.ws().id;
    let phase_rx = crate::faults::phases::watch(if home {
        "direct-enqueued"
    } else {
        "staging-enqueued"
    });
    crate::faults::arm(
        if home {
            "fill-record"
        } else {
            "staging-record"
        },
        1,
    );
    crate::faults::arm("copy-drain", 1);
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(gpu.inner.copy);
    let (at, fill, va, handles) = queue(&set, home);
    phase(phase_rx, "successful H2D enqueue");
    assert!(fill.wait().is_err());
    assert!(matches!(fill.completion(), Filled::Unknown(_)));
    closed(&gpu, &gate);
    set.assert_source();
    same_mapping(&gpu, id, at, va, &handles);
    assert!(set.ws().bind(0, consumer.stream).is_err() && gpu.trim().is_err());
    if !home {
        let b = gpu.inner.staging.buffers.lock().unwrap();
        assert!(!b.held.is_empty() && b.error.is_some());
        assert!(b.held.iter().any(|b| b.last.is_none()));
        println!(
            "VARENA_GPU_WITNESS {{\"kind\":\"staging_no_event\",\"held\":{}}}",
            b.held.len()
        );
    }
    gate.release();
    sync(&gpu, gpu.inner.copy);
    consumer.enqueue(va);
    consumer.verify(&gpu, set.byte);
    gate.check();
}

fn source_handoff() {
    let (gpu, consumer) = setup();
    let mut old = SetFixture::new(&gpu, true, false, 0x5a);
    let new = SetFixture::new(&gpu, true, false, 0x6b);
    let old_id = old.ws().id;
    let enqueued = crate::faults::phases::watch("direct-enqueued");
    let closing = crate::faults::phases::watch("set-close-fences");
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(gpu.inner.copy);
    let (at, old_fill, va, handles) = queue(&old, false);
    phase(enqueued, "old source H2D enqueue");
    old_fill.wait().unwrap();
    let (new_at, new_fill, new_va, _) = queue(&new, false);
    assert!(matches!((at, new_at), (At::Slot(a), At::Slot(b)) if a == b));
    assert_eq!(
        new_va, va,
        "real production slot handoff did not reuse the prepared slot"
    );
    let ws = old.ws.take().unwrap();
    let dropping = std::thread::spawn(move || drop(ws));
    phase(closing, "actual last WeightSet close-fence wait");
    closed(&gpu, &gate);
    old.assert_source();
    same_mapping(&gpu, new.ws().id, at, va, &handles);
    {
        let mut st = gpu.inner.lock();
        let At::Slot(i) = at else { unreachable!() };
        assert_eq!(st.slot(i).holds, Some((new.ws().id, 0)));
        assert!(st
            .set(old_id)
            .pending
            .iter()
            .any(|fill| Arc::ptr_eq(fill, &old_fill)));
    }
    gate.release();
    dropping.join().unwrap();
    new_fill.wait().unwrap();
    sync(&gpu, gpu.inner.copy);
    assert!(old.source.upgrade().is_none() && !old.fd.alive());
    consumer.enqueue(va);
    consumer.verify(&gpu, new.byte);
    // A real later prefetch uses map_home's existing positive-only reap, not a test-only reap.
    new.ws().home(&[0]).unwrap();
    new.ws().prefetch(0).unwrap();
    let next = gpu
        .inner
        .lock()
        .reg(new.ws().id, 0)
        .unwrap()
        .map
        .as_ref()
        .unwrap()
        .fill
        .as_ref()
        .unwrap()
        .clone();
    {
        let mut st = gpu.inner.lock();
        assert!(!st
            .set(new.ws().id)
            .pending
            .iter()
            .any(|fill| Arc::ptr_eq(fill, &new_fill)));
        assert_eq!(st.set(new.ws().id).pending.len(), 1);
    }
    next.wait().unwrap();
    sync(&gpu, gpu.inner.copy);
    gate.check();
}

fn consumer_dependency() {
    let (gpu, consumer) = setup();
    let old = SetFixture::new(&gpu, true, false, 0x5a);
    let new = SetFixture::new(&gpu, true, false, 0x6b);
    let va = warm(&gpu, &old, false);
    let pin = old.ws().bind(0, consumer.stream).unwrap();
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(consumer.stream as CUstream);
    consumer.enqueue(va);
    let marker = gpu
        .inner
        .events
        .record(consumer.stream as CUstream, false)
        .unwrap();
    assert!(!marker.done().unwrap());
    drop(pin); // Actual release event is queued after the real consumer read.
    sync(&gpu, gpu.inner.copy); // Positive producer proof while consumer is gated.
    crate::faults::arm("release-wait", 1);
    crate::faults::arm("release-wait-sync", 1);
    let (at, fill, current_va, handles) = queue(&new, false);
    assert_eq!(current_va, va);
    assert!(fill.wait().is_err());
    assert!(matches!(fill.completion(), Filled::Unknown(_)));
    same_mapping(&gpu, new.ws().id, at, va, &handles);
    assert!(!gpu.inner.custody.lock().unwrap().releases.is_empty());
    assert!(gpu.trim().is_err() && new.ws().bind(0, consumer.stream).is_err());
    old.assert_source();
    gate.release();
    consumer.verify(&gpu, old.byte);
    gate.check();
}

fn staging_prior() {
    let (gpu, consumer) = setup();
    let set = SetFixture::new(&gpu, false, true, 0x5a);
    let g = &gpu.inner;
    // Keep every allocated buffer owned; make one available to deterministically reuse it.
    let pointer = {
        let mut b = g.staging.buffers.lock().unwrap();
        let one = b.free.pop().unwrap();
        let rest = std::mem::take(&mut b.free);
        b.held.extend(rest);
        let pointer = one.ptr as usize;
        b.free.push(one);
        pointer
    };
    let va = g
        .lock()
        .reg(set.ws().id, 0)
        .unwrap()
        .map
        .as_ref()
        .unwrap()
        .va;
    let enqueued = crate::faults::phases::watch("staging-enqueued");
    let _c = CtxGuard::enter(g.ctx).unwrap();
    let gate = Gate::arm(g.copy);
    g.piece(&set.ws().weights, 0, va, 0, BYTES, None).unwrap();
    phase(enqueued, "prior real staging enqueue");
    let event = {
        let b = g.staging.buffers.lock().unwrap();
        assert_eq!(b.free[0].ptr as usize, pointer);
        let event = b.free[0].last.as_ref().unwrap();
        assert!(!event.done().unwrap());
        Arc::downgrade(event)
    };
    crate::faults::arm("staging-prior", 1);
    crate::faults::arm("copy-drain", 1);
    assert!(g.piece(&set.ws().weights, 0, va, 0, BYTES, None).is_err());
    let waiter = gpu.clone();
    let (tx, rx) = mpsc::channel();
    let taking = std::thread::spawn(move || tx.send(waiter.inner.staging.take().is_err()).unwrap());
    assert!(rx.recv_timeout(WAIT).unwrap());
    taking.join().unwrap();
    {
        let b = g.staging.buffers.lock().unwrap();
        assert!(b.free.is_empty());
        assert!(b
            .held
            .iter()
            .any(|b| b.ptr as usize == pointer && b.last.is_some()));
    }
    assert!(event.upgrade().is_some());
    println!("VARENA_GPU_WITNESS {{\"kind\":\"staging_previous_event\",\"ptr\":{pointer}}}");
    set.assert_source();
    closed(&gpu, &gate);
    gate.release();
    sync(&gpu, g.copy);
    consumer.enqueue(va);
    consumer.verify(&gpu, set.byte);
    gate.check();
}

fn last_pin() {
    let (gpu, consumer) = setup();
    let mut set = SetFixture::new(&gpu, true, true, 0x5a);
    let va = warm(&gpu, &set, true);
    let pin = set.ws().bind(0, consumer.stream).unwrap();
    let ws = Arc::downgrade(set.ws());
    let g = Arc::downgrade(&gpu);
    let dropped = crate::faults::phases::watch("pin-drop");
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(consumer.stream as CUstream);
    consumer.enqueue(va);
    let marker = gpu
        .inner
        .events
        .record(consumer.stream as CUstream, false)
        .unwrap();
    assert!(!marker.done().unwrap());
    drop(set.ws.take()); // Public close equivalent: only Pinned now holds this WeightSet Arc.
    crate::faults::arm("pin-release", 1);
    crate::faults::arm("pin-release-drain", 1);
    drop(pin);
    phase(dropped, "actual last Pinned Drop");
    assert!(ws.upgrade().is_some() && g.upgrade().is_some());
    assert_eq!(gpu.inner.custody.lock().unwrap().pins.len(), 1);
    set.assert_source();
    assert!(gpu.trim().is_err());
    gate.release();
    consumer.verify(&gpu, set.byte);
    gate.check();
    drop(gpu);
    assert!(
        g.upgrade().is_some(),
        "actual GPU owner disappeared after its external handle closed"
    );
    println!(
        "VARENA_GPU_WITNESS {{\"kind\":\"last_pin_owners\",\"weight_set\":true,\"gpu\":true}}"
    );
}

fn healthy_drain() {
    let (gpu, consumer) = setup();
    let mut set = SetFixture::new(&gpu, true, true, 0x5a);
    let enqueued = crate::faults::phases::watch("direct-enqueued");
    let draining = crate::faults::phases::watch("copy-drain-enter");
    let failed_job = crate::faults::phases::watch("job-owner-dropped");
    crate::faults::arm("fill-record", 1);
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(gpu.inner.copy);
    let (_, fill, _, _) = queue(&set, true);
    phase(enqueued, "healthy-control actual enqueue");
    phase(draining, "real producer drain entry");
    assert!(fill.state.lock().unwrap().is_none());
    set.assert_source();
    gate.release();
    assert!(fill.wait().is_err());
    assert!(matches!(fill.completion(), Filled::Failed(_)));
    assert!(gpu.inner.check().is_ok());
    phase(failed_job, "failed producer's source-owner release");
    let recovered_job = crate::faults::phases::watch("job-owner-dropped");
    let pin = set.ws().bind(0, consumer.stream).unwrap(); // Actual dead-map recovery and new copy.
    phase(recovered_job, "recovered producer's source-owner release");
    consumer.enqueue(pin.ptr);
    consumer.verify(&gpu, set.byte);
    drop(pin);
    sync(&gpu, consumer.stream as CUstream);
    drop(set.ws.take());
    assert!(set.source.upgrade().is_none() && !set.fd.alive());
    gate.check();
}

fn unregister() {
    let (gpu, consumer) = setup();
    let mut set = SetFixture::new(&gpu, true, true, 0x5a);
    let enqueued = crate::faults::phases::watch("direct-enqueued");
    let closing = crate::faults::phases::watch("set-close-fences");
    crate::faults::arm("set-unregister", 1);
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    let gate = Gate::arm(gpu.inner.copy);
    let (_, fill, va, _) = queue(&set, true);
    phase(enqueued, "actual source enqueue before unregister");
    fill.wait().unwrap();
    let ws = set.ws.take().unwrap();
    let dropping = std::thread::spawn(move || drop(ws));
    phase(closing, "actual WeightSet close before unregister");
    set.assert_source();
    closed(&gpu, &gate);
    gate.release();
    dropping.join().unwrap();
    assert!(gpu.inner.check().is_err());
    set.assert_source();
    consumer.enqueue(va);
    consumer.verify(&gpu, set.byte);
    gate.check();
}

/// The producer is genuinely complete before B admission. Only B's real read is gated;
/// A is distinct and proven complete, so neither the ready shortcut nor A alone can cover B.
fn secondary_consumer(negative: bool) {
    let (gpu, a) = setup();
    let b = Consumer::new(&gpu);
    let mut set = SetFixture::new(&gpu, true, true, 0x5a);
    let id = set.ws().id;
    let produced = crate::faults::phases::watch("job-owner-dropped");
    let va = warm(&gpu, &set, true);
    phase(produced, "secondary producer owner released");
    let pin = set.ws().bind(0, a.stream).unwrap();
    assert_eq!(pin.ptr, va);
    let ready = {
        let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
        pin.ready.done().unwrap()
    };
    assert!(
        ready,
        "producer must be complete before ready-fastpath admission"
    );
    sync(&gpu, a.stream as CUstream);
    let handles = {
        let mut st = gpu.inner.lock();
        st.reg(id, 0)
            .unwrap()
            .map
            .as_ref()
            .unwrap()
            .chunks
            .iter()
            .map(|c| c.handle)
            .collect::<Vec<_>>()
    };
    let ws = Arc::downgrade(set.ws());
    let g = Arc::downgrade(&gpu);
    let dropped = crate::faults::phases::watch("pin-drop");
    let closing = (!negative).then(|| crate::faults::phases::watch("set-close-fences"));
    let _c = CtxGuard::enter(gpu.inner.ctx).unwrap();
    // All streams, registrations, mappings, buffers and event resources now exist.
    let gate = Gate::arm(b.stream as CUstream);
    pin.wait_on(b.stream).unwrap();
    assert_eq!(consumer_ids(&pin), [a.stream, b.stream]);
    assert!(pin.ready.done().unwrap());
    println!("VARENA_GPU_PHASE {{\"phase\":\"secondary_consumer_admitted_ready\"}}");
    b.enqueue(va);
    let pending = gpu
        .inner
        .events
        .record(b.stream as CUstream, false)
        .unwrap();
    assert!(
        !pending.done().unwrap(),
        "gate did not hold the real B read"
    );
    drop(set.ws.take()); // The actual Pinned is now the last external WeightSet owner.
    if negative {
        // The known list is [A,B]. A really records and drains; only B's second record
        // and second fallback drain fail, after its real read was enqueued.
        crate::faults::arm("pin-release", 2);
        crate::faults::arm("pin-release-drain", 2);
        drop(pin);
        phase(dropped, "actual partial all-stream Pinned Drop");
        assert!(!pending.done().unwrap());
        let partial = {
            let custody = gpu.inner.custody.lock().unwrap();
            assert_eq!(
                custody.releases.len(),
                1,
                "actual A release event not retained"
            );
            assert!(
                custody.releases[0].done().unwrap(),
                "A completion not proven"
            );
            assert_eq!(custody.pins.len(), 1);
            assert!(custody.pins.contains_key(&id));
            assert!(custody.error.is_some());
            Arc::downgrade(&custody.releases[0])
        };
        let pins = {
            let mut st = gpu.inner.lock();
            let m = st.reg(id, 0).unwrap().map.as_ref().unwrap();
            assert_eq!(m.pins, 1, "unknown B completion decremented the pin");
            m.pins
        };
        same_mapping(&gpu, id, At::Home, va, &handles);
        set.assert_source();
        assert!(ws.upgrade().is_some() && partial.upgrade().is_some());
        assert!(gpu.trim().is_err());
        {
            // This short-lived upgrade only invokes a real poisoned admission. It is
            // gone before the final last-owner witness and cannot provide custody.
            let actual = ws.upgrade().unwrap();
            assert!(actual.bind(0, b.stream).is_err());
            assert!(actual.prefetch(0).is_err());
        }
        println!("VARENA_GPU_WITNESS {{\"kind\":\"secondary_partial_owners\",\"partial_events\":1,\"pin_count\":{pins},\"weight_set\":true,\"gpu\":true,\"poison_refuses\":true}}");
        gate.release();
        b.verify(&gpu, set.byte);
        gate.check();
        drop(gpu);
        // Actual custody, not test-owned strong clones, keeps the owner graph alive.
        assert!(ws.upgrade().is_some() && g.upgrade().is_some());
        assert!(partial.upgrade().is_some());
        set.assert_source();
    } else {
        let closing = closing.unwrap();
        let dropping = std::thread::spawn(move || drop(pin));
        phase(dropped, "actual all-stream Pinned Drop");
        phase(closing, "actual secondary-consumer WeightSet close");
        // The close phase proves its newly recorded fences exist; A must now be
        // drained again to prove that later A event, independently of gated B.
        sync(&gpu, a.stream as CUstream);
        assert!(
            !dropping.is_finished(),
            "last close completed while B was gated"
        );
        assert!(!pending.done().unwrap());
        {
            let mut st = gpu.inner.lock();
            let m = st.reg(id, 0).unwrap().map.as_mut().unwrap();
            assert_eq!(m.pins, 0, "successful records must release exactly once");
            assert_eq!(m.released.len(), 2);
            assert!(
                m.released[0].done().unwrap(),
                "A must be complete independently"
            );
            assert!(
                !m.released[1].done().unwrap(),
                "B's release fence was bypassed"
            );
            assert!(
                !m.quiet().unwrap(),
                "ready producer and completed A covered gated B"
            );
        }
        assert!(
            gpu.give(BYTES).unwrap().is_empty(),
            "gated B mapping was evicted"
        );
        same_mapping(&gpu, id, At::Home, va, &handles);
        set.assert_source();
        println!("VARENA_GPU_WITNESS {{\"kind\":\"secondary_release_fences\",\"stream_count\":2,\"pin_count\":0,\"release_events\":2,\"producer_ready\":true,\"a_complete\":true,\"b_pending\":true}}");
        // Gate's independent Drop/watchdog releases even if any preceding assertion fails.
        gate.release();
        dropping.join().unwrap();
        b.verify(&gpu, set.byte);
        assert!(set.source.upgrade().is_none() && !set.fd.alive());
        assert!(gpu.inner.lock().sets[id].is_none());
        assert!(gpu.inner.check().is_ok());
        gate.check();
        println!("VARENA_GPU_WITNESS {{\"kind\":\"secondary_close_completed\",\"source_retired\":true,\"mapping_removed\":true}}");
    }
}
