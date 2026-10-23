//! The native GPU suite: the real driver and TensorFS's engine on one rented GPU. Every case is
//! ignored (never the local GPU) and runs in a process of its own (`gpu/native.py`); a case that
//! needs a second process runs this binary again as a role. A case prints `PASS <what> <GPU UUID>`.
#![cfg(target_os = "linux")]

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc};

use tensorfs::io::source::Source;
use tensorfs::io::{self, cuda, Copy, Dst, ItemSource, ReadOpts, Src};
use varena::layout::{Item, Layout, Region, REGION_ALIGN};
use varena::{Access, Backend, CudaBackend, Error, Event, Fill, Fingerprint, GpuId, Level, Memory, Pass, Physical, Pin, Plan, SetId, SetSpec, Step, Va};

/// The role a child run of this binary plays.
const ROLE: &str = "VARENA_GPU_ROLE";

/// A progress line for the runner's watchdog: the case got past this line, so many milliseconds
/// after this process's first step. The runner times each step from them.
macro_rules! step {
    () => {{
        println!("STEP {} {}", line!(), since_first_step());
        std::io::stdout().flush().unwrap();
    }};
}

fn since_first_step() -> u128 {
    static FIRST: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    FIRST.get_or_init(std::time::Instant::now).elapsed().as_millis()
}

/// The one GPU the run was given, and its UUID.
fn gpu() -> (CudaBackend, String) {
    let b = CudaBackend::open(0).expect("a CUDA device");
    step!();
    let mut count = 0;
    step!();
    cuda::check("cuDeviceGetCount", unsafe { (cuda::driver().unwrap().device_get_count)(&mut count) }).unwrap();
    step!();
    assert_eq!(count, 1, "the run must see exactly one GPU");
    step!();
    let uuid = b.gpu().unwrap().uuid;
    step!();
    (b, uuid)
}

/// `len` bytes no two pages share; byte 0 is never 7 (the stray write's value).
fn pattern(len: u64, seed: u8) -> Vec<u8> {
    (0..len).map(|n| (n % 251) as u8 ^ seed ^ 0x80).collect()
}

/// A page of `len` bytes at `va + offset`, filled from `src` through TensorFS, read-only, and
/// its ready event.
fn filled(b: &CudaBackend, va: Va, offset: u64, len: u64, src: Src) -> (Physical, Event) {
    let page = b.create(len).unwrap();
    step!();
    b.map(va, offset, &page, Access::ReadWrite).unwrap();
    step!();
    // SAFETY: the mapping is ours, writable, and nothing else touches it during the read.
    unsafe { io::read(&[Copy { src, dst: b.target(va, offset, len) }], &ReadOpts::default()) }.unwrap();
    step!();
    let ready = b.fence().unwrap();
    step!();
    b.protect(va, offset, len, Access::Read).unwrap();
    step!();
    (page, ready)
}

/// The device bytes at `ptr`, read after `stream`'s work.
fn bytes(ptr: u64, len: u64, stream: usize) -> Vec<u8> {
    let mut out = vec![0; len as usize];
    step!();
    // SAFETY: `ptr` is a live mapping for the whole read; `out` is ours.
    unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr, stream }, dst: Dst::host(&mut out) }], &ReadOpts::default()) }.unwrap();
    step!();
    out
}

fn unload(b: &CudaBackend, va: Va, offset: u64, page: Physical) {
    b.unmap(va, offset, page.len).unwrap();
    step!();
    b.destroy(page).unwrap();
    step!();
}

/// The native runner's GPU monitor: one line per look on stdin, answered `@ <kernels> <memory>`:
/// whether kernels ran, and whether device memory moved, on the run's GPU since the last look
/// (NVML's samples). It makes no CUDA context.
#[test]
#[ignore = "native GPU: the runner's GPU monitor"]
fn role_monitor() {
    if role().as_deref() != Some("monitor") {
        return;
    }
    let mut device = 0;
    cuda::check("cuDeviceGet", unsafe { (cuda::driver().unwrap().device_get)(&mut device, 0) }).unwrap();
    let nvml = platform::gpu::Device::open(&cuda::uuid(device).unwrap()).unwrap();
    let mut seen = 0;
    let _ = nvml.worked_since(&mut seen);
    println!("@ ready");
    std::io::stdout().flush().unwrap();
    for _ in std::io::stdin().lock().lines() {
        // NVML answers "not found" when it has taken no sample since the last look: no signal.
        match nvml.worked_since(&mut seen) {
            Ok((kernels, memory)) => println!("@ {kernels} {memory}"),
            Err(e) => println!("@ unknown {e}"),
        }
        std::io::stdout().flush().unwrap();
    }
}

/// This binary again, running `test` as `role`, with `stdin` (a socket to pass handles on).
fn child(test: &str, role: &str, stdin: Stdio) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", test, "--nocapture", "--test-threads=1"])
        .env(ROLE, role)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap()
}

/// The role this run plays; `None` in a run of the suite itself.
fn role() -> Option<String> {
    std::env::var(ROLE).ok()
}

/// The socket a parent passed as stdin.
fn parent() -> UnixStream {
    UnixStream::from(std::io::stdin().as_fd().try_clone_to_owned().unwrap())
}

fn send(control: &mut UnixStream, handle: &OwnedFd, len: u64) {
    control.write_all(&len.to_le_bytes()).unwrap();
    step!();
    platform::handles::send_fd(control, handle).unwrap();
    step!();
}

fn receive(control: &mut UnixStream) -> (OwnedFd, u64) {
    let mut len = [0; 8];
    step!();
    control.read_exact(&mut len).unwrap();
    step!();
    (platform::handles::recv_fd(control).unwrap(), u64::from_le_bytes(len))
}

/// The first line of `child`'s stdout after `marker`.
fn marked(child: &mut Child, marker: &str) -> String {
    let lines = BufReader::new(child.stdout.take().unwrap()).lines();
    step!();
    lines.map_while(Result::ok).find_map(|l| l.split_once(marker).map(|(_, rest)| rest.to_owned())).unwrap()
}

#[test]
#[ignore = "native GPU: a rental only"]
fn free_is_the_drivers() {
    let (b, uuid) = gpu();
    step!();
    let len = 64 * b.granularity();
    step!();
    let before = b.free().unwrap();
    step!();
    let page = b.create(len).unwrap();
    step!();
    let during = b.free().unwrap();
    step!();
    b.destroy(page).unwrap();
    step!();
    let after = b.free().unwrap();
    step!();
    assert!(before - during >= len, "creating {len} bytes took only {}", before - during);
    step!();
    assert!(after - during >= len, "destroying {len} bytes gave back only {}", after - during);
    step!();
    println!("PASS free is the driver's {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn process_used_counts_this_processes_pages_and_allocations() {
    let (b, uuid) = gpu();
    step!();
    let reading = platform::gpu::Device::open(&uuid).unwrap().read().unwrap();
    step!();
    let how = match reading.processes.is_some_and(|p| p.contains_key(&std::process::id())) {
        true => "NVML per process",
        false => "device delta since open",
    };
    step!();
    let len = 64 * b.granularity();
    step!();
    let start = b.process_used().unwrap();
    step!();
    assert!(start > 0, "this process holds a context");
    step!();
    let context = start;
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let page = b.create(len).unwrap();
    step!();
    b.map(va, 0, &page, Access::ReadWrite).unwrap();
    step!();
    let paged = b.process_used().unwrap();
    step!();
    assert!(paged - start >= len, "a {len} byte page moved process bytes by {}", paged - start);
    step!();
    let pressure = Pressure::take(len);
    step!();
    let allocated = b.process_used().unwrap();
    step!();
    assert!(allocated - paged >= len, "a {len} byte allocation moved process bytes by {}", allocated - paged);
    step!();
    drop(pressure);
    step!();
    unload(&b, va, 0, page);
    step!();
    b.release(va);
    step!();
    let (mut control, theirs) = UnixStream::pair().unwrap();
    step!();
    let mut holder = child("role_holder", "holder", Stdio::from(OwnedFd::from(theirs)));
    step!();
    let (kept, len) = receive(&mut control);
    step!();
    marked(&mut holder, "HOLDER ");
    step!();
    let before = b.process_used().unwrap();
    step!();
    let page = b.import(kept, len).unwrap();
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    b.map(va, 0, &page, Access::Read).unwrap();
    step!();
    let imported = b.process_used().unwrap() as i64 - before as i64;
    step!();
    holder.kill().unwrap();
    step!();
    holder.wait().unwrap();
    step!();
    unload(&b, va, 0, page);
    step!();
    b.release(va);
    step!();
    println!("PASS process bytes ({how}): context {context} at open, pages and allocations counted; a {len} byte imported page moved them {imported} {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn pages_fill_protect_unmap_and_survive_a_clean_owner_exit() {
    let (b, uuid) = gpu();
    step!();
    let len = 2 * b.granularity();
    step!();
    let (first, second) = (pattern(len, 1), pattern(len, 2));
    step!();
    let dir = tempfile::tempdir().unwrap();
    step!();
    let path = dir.path().join("first");
    step!();
    std::fs::write(&path, &first).unwrap();
    step!();
    let file = std::fs::File::open(&path).unwrap();
    step!();
    let va = b.reserve(2 * len).unwrap();
    step!();
    let (p1, r1) = filled(&b, va, 0, len, Src::File(&file, 0));
    step!();
    let (p2, r2) = filled(&b, va, len, len, Src::Host(second.as_ptr()));
    step!();
    let consumer = cuda::Stream::new(0).unwrap();
    step!();
    b.wait(r1, consumer.raw()).unwrap();
    step!();
    b.wait(r2, consumer.raw()).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, consumer.raw()), first);
    step!();
    assert_eq!(bytes(va.base + len, len, consumer.raw()), second);
    step!();
    unload(&b, va, len, p2);
    step!();
    assert_eq!(bytes(va.base, len, consumer.raw()), first, "unmapping one page leaves its neighbour");
    step!();
    let release = b.record(consumer.raw()).unwrap();
    step!();
    consumer.drain().unwrap();
    step!();
    assert!(b.query(release).unwrap());
    step!();
    let kept = b.export(&p1).unwrap();
    step!();
    unload(&b, va, 0, p1);
    step!();
    b.release(va);
    step!();
    for e in [r1, r2, release] {
        b.retire(e);
    }
    drop((consumer, b));
    step!();

    let next = CudaBackend::open(0).unwrap();
    step!();
    let page = next.import(kept, len).unwrap();
    step!();
    let va = next.reserve(len).unwrap();
    step!();
    next.map(va, 0, &page, Access::Read).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, 0), first, "the kept page outlived its owner's exit");
    step!();
    unload(&next, va, 0, page);
    step!();
    next.release(va);
    step!();
    println!("PASS pages fill, protect, unmap and survive a clean owner exit {uuid}");
    step!();
}

/// `stray`: one `st.global.u8` of 7 at its argument. `spin`: busy for `ns` nanoseconds of the
/// GPU's clock (a forward). `copy`: `words` u64s from `src` to `dst`, one thread (a consumer that
/// reads a page).
const KERNELS: &std::ffi::CStr = c"
.version 7.0
.target sm_70
.address_size 64
.visible .entry stray(.param .u64 at) {
    .reg .b64 %a;
    .reg .b32 %v;
    ld.param.u64 %a, [at];
    mov.u32 %v, 7;
    st.global.u8 [%a], %v;
    ret;
}
.visible .entry spin(.param .u64 ns) {
    .reg .b64 %start, %now, %n, %d;
    .reg .pred %p;
    ld.param.u64 %n, [ns];
    mov.u64 %start, %globaltimer;
SPIN:
    mov.u64 %now, %globaltimer;
    sub.u64 %d, %now, %start;
    setp.lt.u64 %p, %d, %n;
    @%p bra SPIN;
    ret;
}
.visible .entry copy(.param .u64 dst, .param .u64 src, .param .u64 words) {
    .reg .b64 %d, %s, %n, %v;
    .reg .pred %p;
    ld.param.u64 %d, [dst];
    ld.param.u64 %s, [src];
    ld.param.u64 %n, [words];
LOOP:
    setp.eq.u64 %p, %n, 0;
    @%p bra DONE;
    ld.global.u64 %v, [%s];
    st.global.u64 [%d], %v;
    add.u64 %s, %s, 8;
    add.u64 %d, %d, 8;
    sub.u64 %n, %n, 1;
    bra LOOP;
DONE:
    ret;
}
";

/// Prepare each diagnostic kernel before a deliberately blocked consumer stream exists.
/// CUDA module/function loading can wait on that stream; doing it inside launch would make
/// the fixture deadlock before it ever exercised Varena's consumer ordering.
fn kernels() -> &'static [usize; 3] {
    static FUNCTIONS: std::sync::OnceLock<[usize; 3]> = std::sync::OnceLock::new();
    FUNCTIONS.get_or_init(|| {
        let d = cuda::driver().unwrap();
        let (_, ctx) = cuda::primary_context(0).unwrap();
        let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
        let mut module = std::ptr::null_mut();
        cuda::check("cuModuleLoadData", unsafe { (d.module_load_data)(&mut module, KERNELS.as_ptr().cast()) }).unwrap();
        [c"stray", c"spin", c"copy"].map(|name| {
            let mut function = std::ptr::null_mut();
            cuda::check("cuModuleGetFunction", unsafe { (d.module_get_function)(&mut function, module, name.as_ptr()) }).unwrap();
            function as usize
        })
    })
}

/// Launches a prepared diagnostic kernel on `stream` with u64 arguments.
fn launch(name: &std::ffi::CStr, args: &[u64], stream: usize) -> cuda::CUresult {
    let index = match name.to_bytes() { b"stray" => 0, b"spin" => 1, b"copy" => 2, _ => panic!("unknown diagnostic kernel") };
    let kernel = kernels()[index] as cuda::CUfunction;
    let d = cuda::driver().unwrap();
    let (_, ctx) = cuda::primary_context(0).unwrap();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let mut values = args.to_vec();
    let mut params: Vec<*mut c_void> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    unsafe { (d.launch_kernel)(kernel, 1, 1, 1, 1, 1, 1, 0, stream as cuda::CUstream, params.as_mut_ptr(), std::ptr::null_mut()) }
}

/// The context's verdict on everything queued so far.
fn synchronize() -> cuda::CUresult {
    let (_, ctx) = cuda::primary_context(0).unwrap();
    step!();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    unsafe { (cuda::driver().unwrap().ctx_synchronize)() }
}

/// Launches the stray write at `ptr` and returns the driver's verdict on it.
fn stray_write(ptr: u64) -> cuda::CUresult {
    match launch(c"stray", &[ptr], 0) {
        cuda::SUCCESS => synchronize(),
        launch => launch,
    }
}

#[test]
#[ignore = "native GPU: a role of read_only_page_faults_a_stray_write_only_in_the_writers_process"]
fn role_stray_writer() {
    if role().as_deref() != Some("writer") {
        return;
    }
    let (b, uuid) = gpu();
    step!();
    let (handle, len) = receive(&mut parent());
    step!();
    let page = b.import(handle, len).unwrap();
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    b.map(va, 0, &page, Access::Read).unwrap();
    step!();
    let code = stray_write(va.base);
    step!();
    let e: Error = cuda::check("the stray write", code).expect_err("a write to a read-only page must fault").into();
    step!();
    assert!(e.is_sticky(), "a read-only fault is sticky: {e}");
    step!();
    println!("FAULT {code} {uuid}");
    step!();
    std::io::stdout().flush().unwrap();
    step!();
    // The context is gone: exit without running destructors against it.
    std::process::exit(0);
}

#[test]
#[ignore = "native GPU: a rental only"]
fn read_only_page_faults_a_stray_write_only_in_the_writers_process() {
    let (b, uuid) = gpu();
    step!();
    let len = b.granularity();
    step!();
    let data = pattern(len, 3);
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    step!();
    let (mut control, theirs) = UnixStream::pair().unwrap();
    step!();
    let mut writer = child("role_stray_writer", "writer", Stdio::from(OwnedFd::from(theirs)));
    step!();
    send(&mut control, &b.export(&page).unwrap(), len);
    step!();
    let fault = marked(&mut writer, "FAULT ");
    step!();
    assert!(writer.wait().unwrap().success());
    step!();
    let (code, theirs) = fault.split_once(' ').unwrap();
    step!();
    assert_eq!(theirs, uuid, "the writer faulted on another GPU");
    step!();
    b.wait(ready, 0).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, 0), data, "the stray write must not land");
    step!();
    let other = b.reserve(len).unwrap();
    step!();
    let (fresh, done) = filled(&b, other, 0, len, Src::Host(data.as_ptr()));
    step!();
    b.wait(done, 0).unwrap();
    step!();
    assert_eq!(bytes(other.base, len, 0), data, "the holder serves on after the writer's fault");
    step!();
    unload(&b, other, 0, fresh);
    step!();
    unload(&b, va, 0, page);
    step!();
    b.release(other);
    step!();
    b.release(va);
    step!();
    println!("PASS a stray write faults only its writer (CUresult {code}) {uuid}");
    step!();
}

/// The device's free bytes, as the driver measures them.
fn driver_free() -> u64 {
    let (_, ctx) = cuda::primary_context(0).unwrap();
    step!();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let (mut free, mut total) = (0usize, 0usize);
    step!();
    cuda::check("cuMemGetInfo", unsafe { (cuda::driver().unwrap().mem_get_info)(&mut free, &mut total) }).unwrap();
    step!();
    free as u64
}

/// Device memory taken with `cuMemAlloc`, as torch takes it.
struct Pressure {
    ctx: cuda::CUcontext,
    taken: Vec<(u64, u64)>,
}

impl Pressure {
    fn none() -> Pressure {
        let (_, ctx) = cuda::primary_context(0).unwrap();
        Pressure { ctx, taken: Vec::new() }
    }

    /// One more allocation of `len` bytes, or the driver's refusal.
    fn try_take(&mut self, len: u64) -> Result<u64, cuda::CUresult> {
        let _ctx = cuda::CtxGuard::enter(self.ctx).unwrap();
        let mut ptr = 0;
        match unsafe { (cuda::driver().unwrap().mem_alloc)(&mut ptr, len as usize) } {
            cuda::SUCCESS => {
                self.taken.push((ptr, len));
                Ok(ptr)
            }
            code => Err(code),
        }
    }

    /// One allocation of `len` bytes.
    fn take(len: u64) -> Pressure {
        let mut p = Pressure::none();
        p.try_take(len).unwrap();
        p
    }

    /// Takes what the device has free, in ever smaller pieces, down to `piece`.
    fn fill(piece: u64) -> Pressure {
        let mut p = Pressure::none();
        let mut size = driver_free();
        while size >= piece {
            match p.try_take(size) {
                Ok(_) => {}
                Err(cuda::ERROR_OUT_OF_MEMORY) => size /= 2,
                Err(code) => panic!("cuMemAlloc: {code}"),
            }
        }
        p
    }

}

impl Drop for Pressure {
    fn drop(&mut self) {
        let _ctx = cuda::CtxGuard::enter(self.ctx).unwrap();
        let d = cuda::driver().unwrap();
        for (ptr, _) in self.taken.drain(..) {
            cuda::check("cuMemFree", unsafe { (d.mem_free)(ptr) }).unwrap();
        }
    }
}

#[test]
#[ignore = "native GPU: a rental only"]
fn cuda_oom_is_recovered_in_process() {
    let (b, uuid) = gpu();
    step!();
    let len = 4 * b.granularity();
    step!();
    let data = pattern(len, 4);
    step!();
    let pressure = Pressure::fill(b.granularity());
    step!();
    let left = b.free().unwrap();
    step!();
    let e = b.create(len).expect_err("a full device refuses a page");
    step!();
    assert!(matches!(e, Error::Device { code: 2, .. }), "expected CUDA_ERROR_OUT_OF_MEMORY, got {e}");
    step!();
    assert!(!e.is_sticky());
    step!();
    drop(pressure);
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    step!();
    b.wait(ready, 0).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, 0), data, "the same process fills after its OOM");
    step!();
    unload(&b, va, 0, page);
    step!();
    b.release(va);
    step!();
    println!("PASS real CUDA OOM recovered in process (free {left} at the OOM) {uuid}");
    step!();
}

/// Holds `stream` in a host callback until dropped.
struct Gate(Option<mpsc::Sender<()>>);

impl Gate {
    fn hold(stream: usize) -> Gate {
        kernels();
        unsafe extern "C" fn held(raw: *mut c_void) {
            // SAFETY: the box `hold` leaked, given to us once.
            let (entered, release) = *unsafe { Box::from_raw(raw.cast::<(mpsc::Sender<()>, mpsc::Receiver<()>)>()) };
            let _ = entered.send(());
            let _ = release.recv();
        }
        let (entered, inside) = mpsc::channel::<()>();
        let (release, wait) = mpsc::channel::<()>();
        let raw = Box::into_raw(Box::new((entered, wait)));
        let (_, ctx) = cuda::primary_context(0).unwrap();
        let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
        let d = cuda::driver().unwrap();
        // Finish lazy kernel preparation too; a zero-length copy touches no address.
        cuda::check("warm diagnostic copy", launch(c"copy", &[0, 0, 0], stream)).unwrap();
        cuda::check("warm diagnostic stream", unsafe { (d.stream_synchronize)(stream as cuda::CUstream) }).unwrap();
        cuda::check("cuLaunchHostFunc", unsafe { (d.launch_host_func)(stream as cuda::CUstream, held, raw.cast()) }).unwrap();
        inside.recv().unwrap();
        Gate(Some(release))
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

#[test]
#[ignore = "native GPU: a rental only"]
fn release_events_order_consumers_without_a_host_wait() {
    let (b, uuid) = gpu();
    step!();
    let len = b.granularity();
    step!();
    let data = pattern(len, 6);
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    step!();
    let consumer = cuda::Stream::new(0).unwrap();
    step!();
    let other = cuda::Stream::new(0).unwrap();
    step!();
    b.wait(ready, consumer.raw()).unwrap();
    step!();
    let gate = Gate::hold(consumer.raw());
    step!();
    let release = b.record(consumer.raw()).unwrap();
    step!();
    b.wait(release, other.raw()).unwrap();
    step!();
    let after = b.record(other.raw()).unwrap();
    step!();
    assert!(!b.query(release).unwrap(), "a pin's release is pending while its consumer runs");
    step!();
    assert!(!b.query(after).unwrap(), "a stream waiting on it waits in-stream");
    step!();
    assert_eq!(b.elapsed(release, after).unwrap(), None);
    step!();
    drop(gate);
    step!();
    other.drain().unwrap();
    step!();
    assert!(b.query(release).unwrap() && b.query(after).unwrap());
    step!();
    assert!(b.elapsed(release, after).unwrap().is_some());
    step!();
    assert_eq!(bytes(va.base, len, other.raw()), data);
    step!();
    for e in [ready, release, after] {
        b.retire(e);
    }
    unload(&b, va, 0, page);
    step!();
    b.release(va);
    step!();
    b.drain().unwrap();
    step!();
    println!("PASS release events order consumers without a host wait {uuid}");
    step!();
}

/// Drops `path`'s pages from the page cache, so the next read comes from the disk.
fn uncached(path: &std::path::Path) {
    let file = std::fs::File::open(path).unwrap();
    step!();
    file.sync_all().unwrap();
    step!();
    assert_eq!(unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
    step!();
}

/// The GB/s of the copy runs TensorFS logged since the last call.
fn rate() -> f64 {
    let runs: Vec<_> = io::metrics::take().into_iter().filter(|c| c.kind == "h2d").collect();
    step!();
    let (bytes, ns) = runs.iter().fold((0, 0), |(b, n), c| (b + c.bytes, n + c.ns));
    step!();
    bytes as f64 / ns.max(1) as f64
}

#[test]
#[ignore = "native GPU: a rental only"]
fn fills_are_reported_against_the_measured_ceiling() {
    let (b, uuid) = gpu();
    step!();
    // A gibibyte: enough for a fill to stand for the link, as TensorFS's own H2D probe is 128 MiB.
    let len = (1u64 << 30).next_multiple_of(b.granularity());
    step!();
    let data = pattern(len, 7);
    step!();
    let dir = tempfile::tempdir().unwrap();
    step!();
    let path = dir.path().join("weights");
    step!();
    std::fs::write(&path, &data).unwrap();
    step!();
    let ceiling = io::ceiling(Some(0), dir.path()).unwrap();
    step!();
    let h2d = ceiling.h2d_gbps.unwrap();
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let mut report = Vec::new();
    step!();
    for (what, limit) in [("ram", h2d), ("ram again", h2d), ("disk", ceiling.disk_gbps.min(h2d))] {
        uncached(&path);
        let file = std::fs::File::open(&path).unwrap();
        let src = if what == "disk" { Src::File(&file, 0) } else { Src::Host(data.as_ptr()) };
        io::metrics::take();
        let (page, ready) = filled(&b, va, 0, len, src);
        let gbps = rate();
        b.wait(ready, 0).unwrap();
        assert_eq!(bytes(va.base, len, 0), data, "{what}");
        unload(&b, va, 0, page);
        b.retire(ready);
        assert!(gbps > 0.0);
        report.push(format!("{what} {gbps:.2} GB/s = {:.0}% of {limit:.2}", 100.0 * gbps / limit));
    }
    b.release(va);
    step!();
    println!("FILLS {} (disk {:.2} GB/s direct={})", report.join("; "), ceiling.disk_gbps, ceiling.disk_direct);
    step!();
    println!("PASS fills reported against the measured ceiling {uuid}");
    step!();
}

/// cuMemSetAccess is documented synchronous: does a map wait for a fill another thread has in flight? Fresh pages
/// mapped with the card idle, then while another thread fills a device buffer from pageable host memory without a
/// pause; each phase's maps split into cuMemMap and cuMemSetAccess (the backend's driver spans). Reported, not judged:
/// the figures say whether varena's mechanism must keep its maps apart from fills.
#[test]
#[ignore = "native GPU: a rental only"]
fn maps_beside_a_fill_in_flight_are_timed_against_maps_on_an_idle_card() {
    time_calls();
    let (b, uuid) = gpu();
    step!();
    let gran = b.granularity();
    // A 1 GiB set's pages per phase: hundreds of maps, enough to see each phase's spread.
    let pages = ((1u64 << 30) / gran) as usize;
    let va = b.reserve(2 * pages as u64 * gran).unwrap();
    step!();
    let made: Vec<Physical> = (0..2 * pages).map(|_| b.create(gran).unwrap()).collect();
    step!();
    let filling = std::sync::atomic::AtomicBool::new(false);
    let stop = std::sync::atomic::AtomicBool::new(false);
    // Each map's milliseconds, of those that began and ended in the phase's state.
    let maps = |half: usize, busy: bool| -> Vec<f64> {
        let mut ms = vec![];
        for (i, page) in made.iter().enumerate().skip(half * pages).take(pages) {
            let before = filling.load(std::sync::atomic::Ordering::Acquire);
            let t = std::time::Instant::now();
            b.map(va, i as u64 * gran, page, Access::ReadWrite).unwrap();
            let took = t.elapsed().as_secs_f64() * 1e3;
            if before == busy && filling.load(std::sync::atomic::Ordering::Acquire) == busy {
                ms.push(took);
            }
        }
        ms.sort_by(f64::total_cmp);
        ms
    };
    let spread = |ms: &[f64]| format!("{} maps, median {:.3} ms, max {:.3} ms", ms.len(), ms.get(ms.len() / 2).copied().unwrap_or(0.0), ms.last().copied().unwrap_or(0.0));
    CALLS.take();
    let idle = maps(0, false);
    step!();
    let idle_calls = CALLS.take();
    println!("TIMINGS idle card: {} | driver {idle_calls}", spread(&idle));
    step!();
    let host = vec![7u8; 1 << 30];
    let buffer = io::DeviceBuffer::new(0, host.len()).unwrap();
    step!();
    let (reads, busy, busy_calls, fill_s) = std::thread::scope(|s| {
        let filler = s.spawn(|| {
            let copy = [Copy { src: Src::Host(host.as_ptr()), dst: Dst::Device { device: 0, ptr: buffer.ptr(), len: host.len(), stream: 0 } }];
            let t = std::time::Instant::now();
            let mut reads = 0u32;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                filling.store(true, std::sync::atomic::Ordering::Release);
                unsafe { io::read(&copy, &ReadOpts::default()) }.unwrap();
                reads += 1;
            }
            filling.store(false, std::sync::atomic::Ordering::Release);
            (reads, t.elapsed().as_secs_f64())
        });
        while !filling.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::yield_now();
        }
        let busy = maps(1, true);
        let calls = CALLS.take();
        stop.store(true, std::sync::atomic::Ordering::Release);
        let (reads, took) = filler.join().unwrap();
        (reads, busy, calls, took)
    });
    step!();
    let gbps = reads as f64 * host.len() as f64 / fill_s / 1e9;
    println!("TIMINGS beside a fill ({reads} reads of 1 GiB at {gbps:.1} GB/s): {} | driver {busy_calls}", spread(&busy));
    step!();
    b.unmap(va, 0, 2 * pages as u64 * gran).unwrap();
    step!();
    made.into_iter().for_each(|p| b.destroy(p).unwrap());
    b.release(va);
    step!();
    println!("PASS maps beside a fill: idle {}; beside a fill {} {uuid}", spread(&idle), spread(&busy));
    step!();
}

// ---- the backend's calls, timed

/// Each backend call's count, total and longest time, by name, and when the last one ended.
#[derive(Default)]
struct Timings(parking_lot::Mutex<std::collections::BTreeMap<&'static str, (u64, std::time::Duration, std::time::Duration)>>, parking_lot::Mutex<std::collections::BTreeMap<&'static str, std::time::Instant>>);

impl Timings {
    const fn new() -> Timings {
        Timings(parking_lot::const_mutex(std::collections::BTreeMap::new()), parking_lot::const_mutex(std::collections::BTreeMap::new()))
    }

    fn add(&self, op: &'static str, took: std::time::Duration) {
        let mut t = self.0.lock();
        let e = t.entry(op).or_default();
        *e = (e.0 + 1, e.1 + took, e.2.max(took));
        self.1.lock().insert(op, std::time::Instant::now());
    }

    fn time<T>(&self, op: &'static str, f: impl FnOnce() -> T) -> T {
        let start = std::time::Instant::now();
        let out = f();
        self.add(op, start.elapsed());
        out
    }

    /// The report, and a fresh start.
    fn take(&self) -> String {
        let report = self.report();
        self.0.lock().clear();
        report
    }

    fn count(&self, op: &str) -> u64 {
        self.0.lock().get(op).map_or(0, |e| e.0)
    }

    /// The wall time since the previous lap (None on the first).
    fn lap(&self) -> Option<std::time::Duration> {
        self.1.lock().insert("lap", std::time::Instant::now()).map(|t| t.elapsed())
    }

    /// `op n total max`, the costliest first.
    fn report(&self) -> String {
        let mut rows: Vec<_> = self.0.lock().iter().map(|(op, (n, total, max))| (*op, *n, *total, *max)).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        rows.iter().map(|(op, n, total, max)| format!("{op} {n}x {total:?} (max {max:?})")).collect::<Vec<_>>().join(", ")
    }
}

/// Every driver call of this process's CUDA backends (their trace spans), by name.
static CALLS: Timings = Timings::new();

/// Times each of the backend's driver spans into [`CALLS`].
struct Calls;

impl<S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>> tracing_subscriber::Layer<S> for Calls {
    fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().replace(std::time::Instant::now());
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if let Some(start) = span.extensions_mut().remove::<std::time::Instant>() {
                CALLS.add(span.metadata().name(), start.elapsed());
            }
        }
    }
}

/// From now on this process times every driver call into [`CALLS`].
fn time_calls() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::Layer;
        let only = tracing_subscriber::filter::Targets::new().with_target("varena::cuda", tracing::Level::TRACE);
        let errors = tracing_subscriber::filter::Targets::new().with_target("varena::memory", tracing::Level::DEBUG);
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(Calls.with_filter(only))
            .with(tracing_subscriber::fmt::layer().with_ansi(false).with_filter(errors))).unwrap();
    });
}

/// The CUDA backend with every call timed: where Memory's time on the device side goes.
struct Timed(CudaBackend, Arc<Timings>);

impl Backend for Timed {
    fn capacity(&self) -> u64 { self.0.capacity() }
    fn headroom(&self) -> u64 { self.0.headroom() }
    fn free(&self) -> varena::Result<u64> { self.1.time("free", || self.0.free()) }
    fn process_used(&self) -> varena::Result<u64> { self.1.time("process_used", || self.0.process_used()) }
    fn opened_context(&self) -> u64 { self.0.opened_context() }
    fn granularity(&self) -> u64 { self.0.granularity() }
    fn reserve(&self, len: u64) -> varena::Result<Va> { self.1.time("reserve", || self.0.reserve(len)) }
    fn release(&self, va: Va) { self.1.time("release", || self.0.release(va)) }
    fn create(&self, len: u64) -> varena::Result<Physical> {
        let page = self.1.time("create", || self.0.create(len));
        // Every 1,000 pages a process says where their time goes (stderr: the case log).
        let n = self.1.count("create");
        if n == 1 || n.is_multiple_of(1000) {
            let wall = self.1.lap();
            eprintln!(
                "TIMINGS pid {} after {n} pages ({wall:?} wall since page {}): backend {} | driver {}",
                std::process::id(),
                n.saturating_sub(1000).max(1),
                self.1.report(),
                CALLS.report()
            );
        }
        page
    }
    fn destroy(&self, page: Physical) -> varena::Result<()> { self.1.time("destroy", || self.0.destroy(page)) }
    fn map(&self, va: Va, offset: u64, page: &Physical, access: Access) -> varena::Result<()> { self.1.time("map", || self.0.map(va, offset, page, access)) }
    fn protect(&self, va: Va, offset: u64, len: u64, access: Access) -> varena::Result<()> { self.1.time("protect", || self.0.protect(va, offset, len, access)) }
    fn unmap(&self, va: Va, offset: u64, len: u64) -> varena::Result<()> { self.1.time("unmap", || self.0.unmap(va, offset, len)) }
    fn export(&self, page: &Physical) -> varena::Result<platform::handles::Owned> { self.1.time("export", || self.0.export(page)) }
    fn import(&self, handle: platform::handles::Owned, len: u64) -> varena::Result<Physical> { self.1.time("import", || self.0.import(handle, len)) }
    fn target(&self, va: Va, offset: u64, len: u64) -> Dst { self.0.target(va, offset, len) }
    fn fence(&self) -> varena::Result<Event> { self.1.time("fence", || self.0.fence()) }
    fn record(&self, stream: usize) -> varena::Result<Event> { self.1.time("record", || self.0.record(stream)) }
    fn query(&self, event: Event) -> varena::Result<bool> { self.1.time("query", || self.0.query(event)) }
    fn elapsed(&self, from: Event, to: Event) -> varena::Result<Option<u64>> { self.1.time("elapsed", || self.0.elapsed(from, to)) }
    fn wait(&self, event: Event, stream: usize) -> varena::Result<()> { self.1.time("wait", || self.0.wait(event, stream)) }
    fn synchronize(&self, stream: usize) -> varena::Result<()> { self.1.time("synchronize", || self.0.synchronize(stream)) }
    fn finish(&self, event: Event) -> varena::Result<()> { self.1.time("finish", || self.0.finish(event)) }
    fn retire(&self, event: Event) { self.1.time("retire", || self.0.retire(event)) }
    fn drain(&self) -> varena::Result<()> { self.1.time("drain", || self.0.drain()) }
}

/// Memory over the timed CUDA backend, and its timings.
fn timed_memory(b: CudaBackend, g: &GpuId) -> (Memory, Arc<Timings>) {
    time_calls();
    let timings = Arc::new(Timings::default());
    (Memory::open(g.clone(), Box::new(Timed(b, timings.clone()))).unwrap(), timings)
}

// ---- Memory over the CUDA backend

/// The run's GPU under a name of this case's own (a machine's TensorD may run beside it); and the
/// real UUID.
fn card(case: &str) -> (CudaBackend, GpuId, String) {
    let (b, uuid) = gpu();
    step!();
    let g = GpuId { uuid: format!("{uuid}-{case}-{}", std::process::id()), ..b.gpu().unwrap() };
    step!();
    (b, g, uuid)
}

/// Block `b` of the set seeded `seed`: a page of bytes of its own.
fn block(seed: u8, b: u32) -> Arc<[u8]> {
    pattern(REGION_ALIGN, seed ^ (b as u8).wrapping_mul(37)).into()
}

/// A set of `blocks` pages, block `b` holding `bytes(b)`, its source inline (host memory).
fn set(name: &str, blocks: u32, fill: Fill, uuid: &str, bytes: impl Fn(u32) -> Arc<[u8]>) -> SetSpec {
    let regions = (0..blocks)
        .map(|b| {
            let at = u64::from(b) * REGION_ALIGN;
            let items = vec![Item { offset: at, len: REGION_ALIGN, source: ItemSource::Inline(bytes(b)), part_end: true }];
            Region { offset: at, nbytes: REGION_ALIGN, span: REGION_ALIGN, items }
        })
        .collect();
    let layout = Layout { parts: vec![], regions, nbytes: u64::from(blocks) * REGION_ALIGN, digest: format!("{name}-{blocks}-{uuid}"), names: vec![] };
    step!();
    let fingerprint = Fingerprint::new(&layout, &[], fill, "release-1", "0/1", uuid);
    step!();
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(Source::files(vec![], false, 0)), fingerprint }
}

/// The bytes every page of a [`bulk`] set holds.
fn bulk_block() -> Arc<[u8]> {
    block(0xb, 0)
}

/// A set of `blocks` pages that all hold one page of bytes: a large set costs one page of RAM.
fn bulk(name: &str, blocks: u32, uuid: &str) -> SetSpec {
    let one = bulk_block();
    step!();
    set(name, blocks, Fill::Stored, uuid, |_| one.clone())
}

/// Bytes the cases leave free for pages (a filler set of more pages than fit takes them all); the
/// rest of the card goes to allocations no one can free. Two gibibytes: room for a one-gibibyte
/// activation to be refused and then made.
const ROOM: u64 = 2 << 30;

/// Takes the card down to [`ROOM`] free, as other allocations of this process would.
fn crowd() -> Pressure {
    let mut p = Pressure::none();
    step!();
    let over = driver_free().saturating_sub(ROOM);
    step!();
    if over > 0 {
        p.try_take(over).unwrap();
    }
    p
}

/// More pages than the card has free bytes: loaded to the GPU, it takes them all.
fn more_than_free() -> u32 {
    (driver_free() / REGION_ALIGN + 1) as u32
}

fn plan(steps: &[&[&str]]) -> Plan {
    Plan { steps: steps.iter().map(|s| Step { sets: s.iter().map(|n| n.to_string()).collect(), passes: 1 }).collect() }
}

fn memory(b: CudaBackend, g: &GpuId) -> Memory {
    time_calls();
    Memory::open(g.clone(), Box::new(b)).unwrap()
}

fn exact(pin: &Pin, want: &[u8], stream: usize) -> bool {
    bytes(pin.ptr, pin.len, stream) == want
}

#[test]
#[ignore = "native GPU: a rental only"]
fn concurrent_binds_of_one_block_fill_it_once() {
    let (b, g, uuid) = card("concurrent");
    step!();
    let mem = memory(b, &g);
    step!();
    let s = mem.load(set("m", 1, Fill::Stored, &g.uuid, |b| block(1, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(s, "k").unwrap();
    step!();
    let streams = [cuda::Stream::new(0).unwrap(), cuda::Stream::new(0).unwrap()];
    step!();
    let retained = mem.retain(s, &[0]).unwrap();
    let pins: Vec<Pin> = std::thread::scope(|t| {
        // Worker threads name their request explicitly; a different thread's active pass
        // is never guessed from process-global state.
        let binds: Vec<_> = streams.iter().map(|st| t.spawn(|| retained.prepare_for(st.raw(), Some(&r)).unwrap().remove(0))).collect();
        binds.into_iter().map(|h| h.join().unwrap()).collect()
    });
    step!();
    for (pin, st) in pins.iter().zip(&streams) {
        assert!(exact(pin, &block(1, 0), st.raw()));
    }
    assert_eq!(pins[0].ptr, pins[1].ptr, "one home");
    step!();
    let moved = r.stats().moved;
    step!();
    assert_eq!(moved.store.bytes, REGION_ALIGN, "one fill for both binders");
    step!();
    drop(pins);
    step!();
    pass.end(0);
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS concurrent binds of one block fill it once {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn an_untransformed_refill_preserves_the_last_consumer_until_preparation_finishes() {
    let (b, g, uuid) = card("release");
    step!();
    let mem = memory(b, &g);
    step!();
    // A declared fill: the bind is fresh, and a pin dropped untransformed sends its page away.
    let s = mem.load(set("t", 1, Fill::Declared("bake-1".into()), &g.uuid, |b| block(2, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["t"]])).unwrap();
    step!();
    let pass = r.pass(s, "k").unwrap();
    step!();
    let consumer = cuda::Stream::new(0).unwrap();
    step!();
    let gate = Gate::hold(consumer.raw());
    step!();
    let first = std::time::Instant::now();
    let pin = mem.bind(s, 0, consumer.raw()).unwrap();
    let first = first.elapsed();
    step!();
    assert!(pin.fresh);
    step!();
    let scratch = Pressure::take(REGION_ALIGN);
    step!();
    let (dst, _) = scratch.taken[0];
    step!();
    assert_eq!(launch(c"copy", &[dst, pin.ptr, REGION_ALIGN / 8], consumer.raw()), cuda::SUCCESS);
    step!();
    drop(pin);
    step!();
    // The copy must read the original contents before the in-place refill lands: the refill's
    // copies wait for it in-stream. The rebind returns once they are issued (D), its page ready
    // in-stream, while the fill thread's read keeps its staging until they complete.
    let _ = mem.classes();
    let (again, waited) = std::thread::scope(|t| {
        let (rebind, waited) = watch(t, first, || mem.bind(s, 0, 0));
        drop(gate);
        (rebind.join().unwrap(), waited)
    });
    step!();
    assert_eq!(synchronize(), cuda::SUCCESS, "the consumer read its page while it was mapped");
    step!();
    assert_eq!(bytes(dst, REGION_ALIGN, 0), *block(2, 0));
    step!();
    assert_eq!(waited, Waited::Returned, "the rebind waited on the host for the gated consumer: its page reads ready at issue ({waited:?})");
    step!();
    let again = again.unwrap();
    step!();
    assert!(again.fresh, "an untransformed page is filled again");
    step!();
    assert!(exact(&again, &block(2, 0), 0));
    step!();
    again.filled();
    step!();
    drop(again);
    step!();
    pass.end(0);
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS an untransformed refill waits during preparation and preserves the old consumer bytes {uuid}");
    step!();
}

/// What a call on another thread was seen doing.
#[derive(Debug, PartialEq)]
enum Waited {
    /// It returned.
    Returned,
    /// Asleep (state S or D) for longer than `budget`, the same call's whole earlier duration:
    /// it waits on the host for something the caller still holds back.
    Asleep,
    /// Running on the CPU for longer than `budget`: it spins on the host, waiting for the device.
    Spun,
}

/// Runs `f` on a thread of `t` and watches it, the caller holding back what `f` may wait for,
/// until `f` returns or is seen waiting on the host (asleep or spinning) for longer than `budget`.
fn watch<'s, T: Send + 's>(t: &'s std::thread::Scope<'s, '_>, budget: std::time::Duration, f: impl FnOnce() -> T + Send + 's) -> (std::thread::ScopedJoinHandle<'s, T>, Waited) {
    let (sent, tid) = mpsc::channel();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = done.clone();
    let handle = t.spawn(move || {
        // The syscall: glibc 2.28 (manylinux_2_28) has no gettid wrapper.
        sent.send(unsafe { libc::syscall(libc::SYS_gettid) }).unwrap();
        let out = f();
        finished.store(true, std::sync::atomic::Ordering::Release);
        out
    });
    let task = format!("/proc/self/task/{}", tid.recv().unwrap());
    let started = std::time::Instant::now();
    loop {
        if done.load(std::sync::atomic::Ordering::Acquire) {
            return (handle, Waited::Returned);
        }
        let (Ok(stat), Ok(sched)) = (std::fs::read_to_string(format!("{task}/stat")), std::fs::read_to_string(format!("{task}/schedstat"))) else {
            // The thread ended between the looks.
            return (handle, Waited::Returned);
        };
        let state = stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().next().map(str::to_owned));
        let ran = sched.split_whitespace().next().unwrap().parse::<u128>().unwrap();
        // Repeated condvar wakes must not reset this diagnostic's observation window.
        // A waiting fill may be notified by unrelated completed work many times.
        if started.elapsed() > budget {
            return (handle, if matches!(state.as_deref(), Some("S" | "D")) || ran < budget.as_nanos() { Waited::Asleep } else { Waited::Spun });
        }
        std::thread::yield_now();
    }
}

/// While a forward runs on the card (a kernel on a stream that reads a set's pages), the bind and
/// pass paths run on the host: binds that evict idle pages, releases, a pass start, the torch cap,
/// an unload, and the backend's own calls (free, process bytes, events). None may wait for the
/// forward: each one's wall time stays below the forward's, which is ten times the bind sequence
/// on an idle card (a call that waited would take at least that long). Every time is reported.
#[test]
#[ignore = "native GPU: a rental only"]
fn the_bind_and_pass_paths_do_not_wait_for_a_running_forward() {
    let (b, g, uuid) = card("forward");
    step!();
    let probe = CudaBackend::open(0).unwrap();
    step!();
    let (mem, timings) = timed_memory(b, &g);
    step!();
    // 64 pages: 128 MiB of binds, each one evicting an idle page on the full card.
    let pages = 64;
    step!();
    let a = mem.load(set("a", 4, Fill::Stored, &g.uuid, |b| block(20, b)), Level::Ram).unwrap();
    step!();
    let (b1, b2) = (mem.load(bulk("b1", pages, &g.uuid), Level::Ram).unwrap(), mem.load(bulk("b2", pages, &g.uuid), Level::Ram).unwrap());
    step!();
    let c = mem.load(set("c", 1, Fill::Stored, &g.uuid, |b| block(21, b)), Level::Ram).unwrap();
    step!();
    let _crowd = crowd();
    mem.load(bulk("idle", more_than_free(), &g.uuid), Level::Gpu).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["a", "b1", "b2", "c"]])).unwrap();
    step!();
    let mut passes: Vec<Pass> = [a, b1, b2].iter().map(|s| r.pass(*s, "k").unwrap()).collect();
    step!();
    let sequence = |set: SetId| {
        let start = std::time::Instant::now();
        let pins: Vec<Pin> = (0..pages).map(|b| mem.bind(set, b, 0).unwrap()).collect();
        drop(pins);
        start.elapsed()
    };
    step!();
    let idle = sequence(b1);
    step!();
    println!("TIMINGS of {pages} binds evicting on an idle card ({idle:?}): {}", timings.report());
    step!();
    println!("TIMINGS of {pages} binds on an idle card ({idle:?}): {}", timings.report());
    step!();
    let forward = idle * 10;
    step!();
    let stream = cuda::Stream::new(0).unwrap();
    step!();
    let other = cuda::Stream::new(0).unwrap();
    step!();
    let pins: Vec<Pin> = (0..4).map(|b| mem.bind(a, b, stream.raw()).unwrap()).collect();
    step!();
    let launched = std::time::Instant::now();
    step!();
    assert_eq!(launch(c"spin", &[forward.as_nanos() as u64], stream.raw()), cuda::SUCCESS);
    step!();
    drop(pins);
    step!();
    let mut times: Vec<(&str, std::time::Duration)> = vec![];
    let mut running = false;
    step!();
    let mut time = |what: &'static str, f: &mut dyn FnMut()| {
        let start = std::time::Instant::now();
        f();
        times.push((what, start.elapsed()));
    };
    step!();
    time("binds+evictions", &mut || {
        sequence(b2);
    });
    step!();
    time("pass start", &mut || passes.push(r.pass(c, "k").unwrap()));
    step!();
    time("torch_limit", &mut || {
        mem.classes();
    });
    step!();
    time("unload", &mut || mem.unload(b1));
    step!();
    time("free", &mut || {
        probe.free().unwrap();
    });
    step!();
    time("process_used", &mut || {
        let _ = probe.process_used();
    });
    step!();
    time("record+query+wait", &mut || {
        let e = probe.record(stream.raw()).unwrap();
        running = !probe.query(e).unwrap();
        probe.wait(e, other.raw()).unwrap();
        probe.retire(e);
    });
    step!();
    stream.drain().unwrap();
    step!();
    let ran = launched.elapsed();
    step!();
    assert!(ran >= forward, "the forward ran {ran:?}, asked {forward:?}");
    step!();
    let report: Vec<String> = times.iter().map(|(w, t)| format!("{w} {t:?}")).collect();
    step!();
    println!("FORWARD {ran:?} (binds+evictions of {pages} pages on an idle card: {idle:?}), still running at the last call: {running}: {}", report.join(", "));
    println!("TIMINGS in all: {}", timings.report());
    step!();
    for (what, took) in &times {
        assert!(*took < forward, "{what} took {took:?} under a {forward:?} forward: it waited for it");
    }
    drop(passes);
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS the bind and pass paths do not wait for a running forward {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_page_larger_than_the_card_is_exhausted_before_any_io_and_its_id_serves_again() {
    let (b, g, uuid) = card("huge");
    step!();
    let span = (b.capacity() + REGION_ALIGN).next_multiple_of(REGION_ALIGN);
    step!();
    let mem = memory(b, &g);
    step!();
    let one: Arc<[u8]> = vec![7u8].into();
    step!();
    let items = vec![Item { offset: 0, len: 1, source: ItemSource::Inline(one), part_end: true }];
    step!();
    let layout = Layout { parts: vec![], regions: vec![Region { offset: 0, nbytes: span, span, items }], nbytes: span, digest: format!("huge-{}", g.uuid), names: vec![] };
    step!();
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", &g.uuid);
    step!();
    let source = Arc::new(Source::files(vec![], false, 0));
    step!();
    // To the GPU: it maps only into free bytes, so the page is never filled.
    let huge = mem.load(SetSpec { name: "huge".into(), layout: Arc::new(layout), source, fingerprint }, Level::Gpu).unwrap();
    step!();
    io::metrics::take();
    step!();
    let r = mem.request("r", plan(&[&["huge"]])).unwrap();
    step!();
    let pass = r.pass(huge, "k");
    step!();
    let e = match &pass {
        Err(e) => e.clone(),
        Ok(_) => mem.bind(huge, 0, 0).err().expect("a page larger than the card cannot bind"),
    };
    step!();
    assert!(matches!(e, Error::Exhausted(_)), "{e}");
    step!();
    drop(pass);
    step!();
    assert!(io::metrics::take().iter().all(|c| c.bytes == 0), "no byte moved before the refusal");
    step!();
    let stats = r.end();
    step!();
    assert_eq!((stats.leftover, stats.moved.store.bytes), (0, 0), "{stats:?}");
    step!();
    let m = mem.load(set("m", 1, Fill::Stored, &g.uuid, |b| block(4, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "k").unwrap();
    step!();
    let pin = mem.bind(m, 0, 0).unwrap();
    step!();
    assert!(exact(&pin, &block(4, 0), 0), "the same request id serves again");
    step!();
    drop(pin);
    step!();
    pass.end(0);
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS a page larger than the card is exhausted before any I/O, and its id serves again {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_bind_on_a_full_card_takes_idle_pages_then_is_exhausted_and_recovers_when_room_returns() {
    let (b, g, uuid) = card("full");
    step!();
    let mem = memory(b, &g);
    step!();
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(5, b)), Level::Ram).unwrap();
    step!();
    mem.load(set("idle", 1, Fill::Stored, &g.uuid, |b| block(6, b)), Level::Gpu).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "k").unwrap();
    step!();
    // Everything else on the card is an allocation varena cannot free.
    let full = Pressure::fill(64 << 10);
    step!();
    let first = mem.bind(m, 0, 0).expect("the idle page makes room");
    step!();
    let e = mem.bind(m, 1, 0).err().expect("nothing is left to free");
    step!();
    assert!(matches!(e, Error::Exhausted(_)), "{e}");
    step!();
    drop(full);
    step!();
    let second = mem.bind(m, 1, 0).expect("room came back: the same request binds");
    step!();
    assert!(exact(&first, &block(5, 0), 0) && exact(&second, &block(5, 1), 0));
    step!();
    drop((first, second));
    step!();
    pass.end(0);
    step!();
    let stats = r.end();
    step!();
    assert!(stats.evicted.iter().sum::<u64>() >= REGION_ALIGN && stats.leftover == 0, "{stats:?}");
    step!();
    println!("PASS a bind on a full card takes idle pages, then is exhausted, and recovers when room returns {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn repeated_passes_of_a_set_keep_its_declared_step() {
    let (b, g, uuid) = card("steps");
    step!();
    let mem = memory(b, &g);
    step!();
    let names = ["a", "vae", "audio", "dit", "lora"];
    step!();
    let ids: Vec<SetId> = names.iter().zip(1u8..).map(|(n, s)| mem.load(set(n, 1, Fill::Stored, &g.uuid, |b| block(s, b)), Level::Ram).unwrap()).collect();
    step!();
    let id = |n: &str| ids[names.iter().position(|x| *x == n).unwrap()];
    step!();
    let r = mem.request("r", plan(&[&["a"], &["vae", "audio"], &["dit", "lora"], &["audio"], &["vae"]])).unwrap();
    step!();
    let forward = |n: &str| {
        let pass = r.pass(id(n), "k").unwrap();
        drop(mem.bind(id(n), 0, 0).unwrap());
        pass.end(0);
        r.stats().trace.last().map(|t| t.step).unwrap()
    };
    step!();
    assert_eq!(forward("a"), 0);
    step!();
    assert_eq!(forward("audio"), 1);
    step!();
    for _ in 0..3 {
        assert_eq!(forward("vae"), 1, "a repeated encode keeps its step, not the decode's");
    }
    assert_eq!((forward("dit"), forward("lora")), (2, 2));
    step!();
    assert_eq!((forward("audio"), forward("vae")), (3, 4));
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS repeated passes of a set keep its declared step {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_transform_binds_another_set_under_pressure_and_a_second_binder_waits_for_it() {
    let (b, g, uuid) = card("transform");
    step!();
    let mem = memory(b, &g);
    step!();
    let base = mem.load(set("base", 1, Fill::Declared("lora-1".into()), &g.uuid, |b| block(7, b)), Level::Ram).unwrap();
    step!();
    let factors = mem.load(set("factors", 1, Fill::Stored, &g.uuid, |b| block(8, b)), Level::Ram).unwrap();
    step!();
    // The stream first: on the full card a stream's own memory would not fit.
    let stream = cuda::Stream::new(0).unwrap();
    step!();
    let _crowd = crowd();
    mem.load(bulk("idle", more_than_free(), &g.uuid), Level::Gpu).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["base", "factors"]])).unwrap();
    step!();
    let pass = r.pass(base, "k").unwrap();
    step!();
    let first = std::time::Instant::now();
    let pin = mem.bind(base, 0, stream.raw()).unwrap();
    let first = first.elapsed();
    step!();
    assert!(pin.fresh, "the caller bakes it");
    step!();
    // The bake binds the factors on the card the idle set fills, and copies them in.
    let f = mem.bind(factors, 0, stream.raw()).unwrap();
    step!();
    assert_eq!(launch(c"copy", &[pin.ptr, f.ptr, REGION_ALIGN / 8], stream.raw()), cuda::SUCCESS);
    step!();
    let (second, waited) = std::thread::scope(|t| {
        let (waiter, waited) = watch(t, first, || mem.bind(base, 0, 0).unwrap());
        pin.filled();
        (waiter.join().unwrap(), waited)
    });
    step!();
    assert_eq!(waited, Waited::Asleep, "the second binder must wait, asleep, for the bake ({waited:?})");
    step!();
    assert!(!second.fresh, "the second binder gets the baked page");
    step!();
    assert!(exact(&second, &block(8, 0), 0), "never a half-made page");
    step!();
    drop((pin, f, second));
    step!();
    pass.end(0);
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS a transform binds another set under pressure, and a second binder waits for it {uuid}");
    step!();
}

// ---- several processes

