//! The native GPU suite: the real driver and TensorFS's engine on one rented GPU. Every case is
//! ignored (never the local GPU) and runs in a process of its own (`gpu/native.py`); a case that
//! needs a second process runs this binary again as a role. A case prints `PASS <what> <GPU UUID>`.
#![cfg(target_os = "linux")]

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, Stdio};
use std::collections::HashMap;
use std::sync::{mpsc, Arc};

use tensorfs::io::source::Source;
use tensorfs::io::{self, cuda, Copy, Dst, ItemSource, ReadOpts, Src};
use varena::layout::{Item, Layout, Region, REGION_ALIGN};
use varena::{Access, Backend, CudaBackend, Error, Event, Fill, Fingerprint, GpuId, Ledger, Level, Measured, Memory, Next, Pass, Physical, Pin, Plan, Request, SetId, SetSpec, Stats, Step, Torch, Va};

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
    b.destroy(page);
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
    b.destroy(page);
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

/// Launches kernel `name` of [`KERNELS`] on `stream` with `args` (u64s).
fn launch(name: &std::ffi::CStr, args: &[u64], stream: usize) -> cuda::CUresult {
    let d = cuda::driver().unwrap();
    step!();
    let (_, ctx) = cuda::primary_context(0).unwrap();
    step!();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let (mut module, mut kernel) = (std::ptr::null_mut(), std::ptr::null_mut());
    step!();
    cuda::check("cuModuleLoadData", unsafe { (d.module_load_data)(&mut module, KERNELS.as_ptr().cast()) }).unwrap();
    step!();
    cuda::check("cuModuleGetFunction", unsafe { (d.module_get_function)(&mut kernel, module, name.as_ptr()) }).unwrap();
    step!();
    let mut values = args.to_vec();
    step!();
    let mut params: Vec<*mut c_void> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    step!();
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

    fn bytes(&self) -> u64 {
        self.taken.iter().map(|(_, n)| n).sum()
    }

    /// Torch's figures, were these its allocations.
    fn torch(&self) -> Torch {
        Torch { allocated: self.bytes(), reserved: self.bytes() }
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

#[test]
#[ignore = "native GPU: a role of kept_page_is_imported_bit_exact_after_its_holders_kill_9"]
fn role_holder() {
    if role().as_deref() != Some("holder") {
        return;
    }
    let (b, uuid) = gpu();
    step!();
    let len = 4 * b.granularity();
    step!();
    let data = pattern(len, 5);
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    step!();
    b.wait(ready, 0).unwrap();
    step!();
    let mut control = parent();
    step!();
    send(&mut control, &b.export(&page).unwrap(), len);
    step!();
    println!("HOLDER {uuid}");
    step!();
    std::io::stdout().flush().unwrap();
    step!();
    // Held until killed.
    control.read_exact(&mut [0]).unwrap();
    step!();
    unreachable!("the holder is killed, never let go");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn kept_page_is_imported_bit_exact_after_its_holders_kill_9() {
    let (mut control, theirs) = UnixStream::pair().unwrap();
    step!();
    let mut holder = child("role_holder", "holder", Stdio::from(OwnedFd::from(theirs)));
    step!();
    let (kept, len) = receive(&mut control);
    step!();
    let uuid = marked(&mut holder, "HOLDER ");
    step!();
    holder.kill().unwrap();
    step!();
    assert_eq!(holder.wait().unwrap().signal(), Some(libc::SIGKILL));
    step!();
    let (b, mine) = gpu();
    step!();
    assert_eq!(mine, uuid);
    step!();
    let page = b.import(kept, len).unwrap();
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    b.map(va, 0, &page, Access::Read).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, 0), pattern(len, 5), "the kept page is the holder's, bit-exact");
    step!();
    let before = b.free().unwrap();
    step!();
    unload(&b, va, 0, page);
    step!();
    b.release(va);
    step!();
    let after = b.free().unwrap();
    step!();
    assert!(after - before >= len, "the last reference gone, its {len} bytes are free (got {})", after - before);
    step!();
    println!("PASS a kept page is imported bit-exact after its holder's kill -9 {uuid}");
    step!();
}

/// Holds `stream` in a host callback until dropped.
struct Gate(Option<mpsc::Sender<()>>);

impl Gate {
    fn hold(stream: usize) -> Gate {
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

/// A RAM fill read again. Unregistered, TensorFS's engine stages a range's first read, then
/// registers it for the length of each later read (cuMemHostRegister, then cuMemHostUnregister as
/// the read ends): every reuse pays the registration. Registered once by its owner through
/// `Backend::register_host` (as the executor does when it maps a RAM-tier layout), every read
/// copies directly. Rates are reported as % of TensorFS's measured H2D ceiling.
#[test]
#[ignore = "native GPU: a rental only"]
fn a_registered_host_range_fills_directly_every_time() {
    let (b, uuid) = gpu();
    step!();
    let (h2d, _) = io::metrics::h2d(0).unwrap();
    step!();
    let len = (1u64 << 30).next_multiple_of(b.granularity());
    step!();
    let data = pattern(len, 9);
    step!();
    let va = b.reserve(len).unwrap();
    step!();
    let page = b.create(len).unwrap();
    step!();
    b.map(va, 0, &page, Access::ReadWrite).unwrap();
    step!();
    let mut rows = vec![];
    step!();
    let mut read = |what: &str| -> (f64, u64, u64) {
        io::metrics::take();
        // SAFETY: the mapping is ours and writable; `data` outlives the read.
        unsafe { io::read(&[Copy { src: Src::Host(data.as_ptr()), dst: b.target(va, 0, len) }], &ReadOpts::default()) }.unwrap();
        let runs: Vec<_> = io::metrics::take().into_iter().filter(|c| c.kind == "h2d").collect();
        let sum = |f: fn(&io::metrics::Copy) -> u64| runs.iter().map(f).sum::<u64>();
        let gbps = sum(|c| c.bytes) as f64 / sum(|c| c.ns).max(1) as f64;
        let (pin, staged) = (sum(|c| c.pin_ns), sum(|c| c.staged));
        rows.push(format!("{what} {gbps:.2} GB/s = {:.0}% (pin {:.0} ms, staged {} MiB)", 100.0 * gbps / h2d, pin as f64 / 1e6, staged >> 20));
        (gbps, pin, staged)
    };
    step!();
    read("unregistered 1");
    step!();
    let (again, ..) = read("unregistered 2");
    step!();
    read("unregistered 3");
    step!();
    // `data` stays mapped until it is unregistered below.
    b.register_host(data.as_ptr() as u64, len).unwrap();
    step!();
    let first = read("registered 1");
    step!();
    let second = read("registered 2");
    step!();
    b.unregister_host(data.as_ptr() as u64).unwrap();
    step!();
    assert_eq!(bytes(va.base, len, 0), data);
    step!();
    b.unmap(va, 0, len).unwrap();
    step!();
    b.destroy(page);
    step!();
    b.release(va);
    step!();
    println!("REUSE of {h2d:.2} GB/s: {}", rows.join("; "));
    step!();
    for (gbps, pin, staged) in [first, second] {
        assert_eq!((pin, staged), (0, 0), "a registered range copies directly, never staged or pinned per read");
        assert!(gbps > again, "a registered read beats an unregistered reuse ({gbps:.2} vs {again:.2} GB/s)");
    }
    println!("PASS a registered host range fills directly every time (second {:.0}% vs first {:.0}%) {uuid}", 100.0 * second.0 / h2d, 100.0 * first.0 / h2d);
    step!();
}

// ---- Memory over the CUDA backend

/// The run's GPU under a name of this case's own, so its table, custody and kept pages are the
/// case's (a machine's TensorD may run beside it); and the real UUID.
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
    Memory::open(g.clone(), None, Box::new(b)).unwrap()
}

fn exact(pin: &Pin, want: &[u8], stream: usize) -> bool {
    bytes(pin.ptr, pin.len, stream) == want
}

/// An activation of `len` bytes taken as torch takes one: on each real OOM, `r.oom` answers. The
/// design bounds the answers: pages + cut levels + 2. Returns them.
fn activation(r: &Request, set: SetId, p: &mut Pressure, len: u64, pages: u32) -> Vec<String> {
    let mut answers = vec![];
    step!();
    while let Err(code) = p.try_take(len) {
        assert_eq!(code, cuda::ERROR_OUT_OF_MEMORY);
        let next = r.oom(Some(set), Some(len), None, p.torch());
        answers.push(format!("{next:?}").split('(').next().unwrap_or_default().to_string());
        assert!(matches!(next, Next::Retry), "idle pages answer Retry: {answers:?}");
        assert!(answers.len() <= pages as usize + 2, "the OOM ladder must end: {answers:?}");
    }
    answers
}

#[test]
#[ignore = "native GPU: a rental only"]
fn concurrent_binds_of_one_block_fill_it_once() {
    let (b, g, uuid) = card("concurrent");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let s = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(1, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(s, "main", "1x1", Torch::default()).unwrap();
    step!();
    let streams = [cuda::Stream::new(0).unwrap(), cuda::Stream::new(0).unwrap()];
    step!();
    let pins: Vec<Pin> = std::thread::scope(|t| {
        let binds: Vec<_> = streams.iter().map(|st| t.spawn(|| mem.bind(s, 0, st.raw()).unwrap())).collect();
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
    assert_eq!(moved.store.bytes + moved.tier.bytes, REGION_ALIGN, "one fill for both binders");
    step!();
    drop(pins);
    step!();
    pass.end(Measured::default(), Torch::default());
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS concurrent binds of one block fill it once {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_rebind_waits_in_stream_while_the_last_consumer_runs() {
    let (b, g, uuid) = card("release");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    // A declared fill: the bind is fresh, and a pin dropped untransformed sends its page away.
    let s = mem.load(set("t", 1, Fill::Declared("bake-1".into()), &g.uuid, |b| block(2, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["t"]])).unwrap();
    step!();
    let pass = r.pass(s, "main", "1x1", Torch::default()).unwrap();
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
    // Memory takes the release now; its event waits behind the gate and the copy. The rebind
    // refills the page in place after it: in-stream, never waiting for it on the host.
    let _ = mem.torch_limit();
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
    assert_eq!(waited, Waited::Returned, "the rebind waited on the host while the last consumer ran (longer than a whole first bind took)");
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
    pass.end(Measured::default(), Torch::default());
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS a rebind waits in-stream while the last consumer runs {uuid}");
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
    let mut asleep: Option<std::time::Instant> = None;
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
        if matches!(state.as_deref(), Some("S" | "D")) {
            if asleep.get_or_insert_with(std::time::Instant::now).elapsed() > budget {
                return (handle, Waited::Asleep);
            }
        } else {
            asleep = None;
        }
        if ran > budget.as_nanos() {
            return (handle, Waited::Spun);
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
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
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
    let mut passes: Vec<Pass> = [a, b1, b2].iter().map(|s| r.pass(*s, "main", "1", Torch::default()).unwrap()).collect();
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
    time("pass start", &mut || passes.push(r.pass(c, "main", "1", Torch::default()).unwrap()));
    step!();
    time("torch_limit", &mut || {
        mem.torch_limit();
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
        assert!(!probe.query(e).unwrap(), "the forward is still running");
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
    println!("FORWARD {ran:?} (binds+evictions of {pages} pages on an idle card: {idle:?}): {}", report.join(", "));
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

/// A pass of `m` (two pinned blocks) whose one-gibibyte activation meets a real OOM: the card is
/// crowded and the rest full of an idle set. Returns the OOM answers and the record.
fn first_run(mem: &Memory, m: SetId, idle: u32) -> (Vec<String>, Stats) {
    let r = mem.request("first", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "main", "1x8", Torch::default()).unwrap();
    step!();
    let pins: Vec<Pin> = (0..2).map(|b| mem.bind(m, b, 0).unwrap()).collect();
    step!();
    let mut p = Pressure::none();
    step!();
    let answers = activation(&r, m, &mut p, ROOM / 2, idle);
    step!();
    for (b, pin) in pins.iter().enumerate() {
        assert!(exact(pin, &block(3, b as u32), 0), "pins outlive the OOM");
    }
    pass.end(Measured { allocated_peak: ROOM / 2, slack: 0, sites: vec![] }, p.torch());
    step!();
    drop((pins, p));
    step!();
    (answers, r.end())
}

#[test]
#[ignore = "native GPU: a rental only"]
fn an_oom_in_a_pass_is_recovered_in_process() {
    let (b, g, uuid) = card("oom");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(3, b)), Level::Ram).unwrap();
    step!();
    let _crowd = crowd();
    let idle = more_than_free();
    step!();
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    step!();
    let (answers, stats) = first_run(&mem, m, idle);
    step!();
    assert!(!answers.is_empty(), "the full card refused the activation");
    step!();
    assert!(stats.retries >= 1 && stats.evicted.iter().sum::<u64>() >= ROOM / 2, "{stats:?}");
    step!();
    assert_eq!(stats.leftover, 0);
    step!();
    println!("PASS an OOM in a pass is recovered in process (answers {answers:?}) {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_measured_pass_gets_its_room_before_it_runs() {
    let (b, g, uuid) = card("room");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(3, b)), Level::Ram).unwrap();
    step!();
    let _crowd = crowd();
    let idle = more_than_free();
    step!();
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    step!();
    let (answers, _) = first_run(&mem, m, idle);
    step!();
    assert!(!answers.is_empty(), "the first run met a real OOM");
    step!();
    // The idle set takes the card again; the second run's pass makes its measured room first.
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    step!();
    assert!(driver_free() < ROOM / 2, "the card is full before the second pass: {} free", driver_free());
    step!();
    let r = mem.request("second", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "main", "1x8", Torch::default()).unwrap();
    step!();
    let mut p = Pressure::none();
    step!();
    assert_eq!(p.try_take(ROOM / 2).map(|_| ()), Ok(()), "the pass made its measured room before it ran");
    step!();
    pass.end(Measured { allocated_peak: ROOM / 2, slack: 0, sites: vec![] }, p.torch());
    step!();
    let stats = r.end();
    step!();
    assert_eq!(stats.retries, 0, "{stats:?}");
    step!();
    println!("PASS a measured pass gets its room before it runs {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_page_larger_than_the_card_is_exhausted_before_any_io_and_its_id_serves_again() {
    let (b, g, uuid) = card("huge");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
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
    // To the GPU: it maps only into free bytes, so the page is never filled (the RAM tier would be).
    let huge = mem.load(SetSpec { name: "huge".into(), layout: Arc::new(layout), source, fingerprint }, Level::Gpu).unwrap();
    step!();
    io::metrics::take();
    step!();
    let r = mem.request("r", plan(&[&["huge"]])).unwrap();
    step!();
    let pass = r.pass(huge, "main", "1", Torch::default());
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
    assert_eq!((stats.leftover, stats.moved.store.bytes + stats.moved.tier.bytes), (0, 0), "{stats:?}");
    step!();
    let m = mem.load(set("m", 1, Fill::Stored, &g.uuid, |b| block(4, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "main", "1", Torch::default()).unwrap();
    step!();
    let pin = mem.bind(m, 0, 0).unwrap();
    step!();
    assert!(exact(&pin, &block(4, 0), 0), "the same request id serves again");
    step!();
    drop(pin);
    step!();
    pass.end(Measured::default(), Torch::default());
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
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(5, b)), Level::Ram).unwrap();
    step!();
    mem.load(set("idle", 1, Fill::Stored, &g.uuid, |b| block(6, b)), Level::Gpu).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    step!();
    let pass = r.pass(m, "main", "1", Torch::default()).unwrap();
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
    pass.end(Measured::default(), Torch::default());
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
    let _ledger = Ledger::create(&g, None).unwrap();
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
        let pass = r.pass(id(n), "main", "1", Torch::default()).unwrap();
        drop(mem.bind(id(n), 0, 0).unwrap());
        pass.end(Measured::default(), Torch::default());
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
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let base = mem.load(set("base", 1, Fill::Declared("lora-1".into()), &g.uuid, |b| block(7, b)), Level::Ram).unwrap();
    step!();
    let factors = mem.load(set("factors", 1, Fill::Stored, &g.uuid, |b| block(8, b)), Level::Ram).unwrap();
    step!();
    let _crowd = crowd();
    mem.load(bulk("idle", more_than_free(), &g.uuid), Level::Gpu).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["base", "factors"]])).unwrap();
    step!();
    let pass = r.pass(base, "main", "1", Torch::default()).unwrap();
    step!();
    let stream = cuda::Stream::new(0).unwrap();
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
    pass.end(Measured::default(), Torch::default());
    step!();
    assert_eq!(r.end().leftover, 0);
    step!();
    println!("PASS a transform binds another set under pressure, and a second binder waits for it {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_ram_level_set_binds_from_the_tier() {
    let (b, g, uuid) = card("tier");
    step!();
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    step!();
    let s = mem.load(set("t", 4, Fill::Stored, &g.uuid, |b| block(9, b)), Level::Ram).unwrap();
    step!();
    let r = mem.request("r", plan(&[&["t"]])).unwrap();
    step!();
    let pass = r.pass(s, "main", "1", Torch::default()).unwrap();
    step!();
    let pins: Vec<Pin> = (0..4).map(|b| mem.bind(s, b, 0).unwrap()).collect();
    step!();
    for (b, pin) in pins.iter().enumerate() {
        assert!(exact(pin, &block(9, b as u32), 0));
    }
    drop(pins);
    step!();
    pass.end(Measured::default(), Torch::default());
    step!();
    let stats = r.end();
    step!();
    let origins: Vec<String> = stats.trace.iter().map(|t| format!("{:?}", t.origin)).collect();
    step!();
    assert_eq!(stats.moved.tier.bytes, 4 * REGION_ALIGN, "{origins:?}");
    step!();
    let gbps = stats.moved.tier.bytes as f64 / stats.moved.tier.ns.max(1) as f64;
    step!();
    println!("FILLS tier {gbps:.2} GB/s ({origins:?})");
    step!();
    println!("PASS a RAM-level set binds from the tier {uuid}");
    step!();
}

// ---- several processes

/// A child's Memory over the CUDA backend, one command per stdin line; each reply is a line
/// `@ <reply>` on stdout.
#[test]
#[ignore = "native GPU: the agent of the multi-process cases"]
fn role_agent() {
    if role().as_deref() != Some("agent") {
        return;
    }
    let mut mem: Option<Memory> = None;
    step!();
    let mut uuid = String::new();
    step!();
    let mut sets: HashMap<String, SetId> = HashMap::new();
    step!();
    let mut requests: HashMap<String, Request> = HashMap::new();
    step!();
    let mut passes: HashMap<String, Pass> = HashMap::new();
    step!();
    let mut pins: Vec<Pin> = vec![];
    step!();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        let m = || mem.as_ref().unwrap();
        let bytes_of = |seed: &str, b: u32| if seed == "bulk" { bulk_block() } else { block(seed.parse().unwrap(), b) };
        let reply = match w.as_slice() {
            ["open", id] => {
                let (b, real) = gpu();
                uuid = id.to_string();
                let g = GpuId { uuid: uuid.clone(), ..b.gpu().unwrap() };
                mem = Some(memory(b, &g));
                format!("ok {} {real}", std::process::id())
            }
            ["load", name, blocks, seed, fill, level] => {
                let blocks = if *blocks == "more" { more_than_free() } else { blocks.parse().unwrap() };
                let fill = match *fill {
                    "stored" => Fill::Stored,
                    other => Fill::Declared(other.into()),
                };
                let level = if *level == "gpu" { Level::Gpu } else { Level::Ram };
                match m().load(set(name, blocks, fill, &uuid, |b| bytes_of(seed, b)), level) {
                    Ok(id) => {
                        sets.insert(name.to_string(), id);
                        "ok".into()
                    }
                    Err(e) => format!("err {e}"),
                }
            }
            ["request", id, steps] => {
                let steps: Vec<Vec<&str>> = steps.split(';').map(|s| s.split(',').collect()).collect();
                let steps: Vec<&[&str]> = steps.iter().map(Vec::as_slice).collect();
                requests.insert(id.to_string(), m().request(id, plan(&steps)).unwrap());
                "ok".into()
            }
            ["pass", req, set, shape] => match requests[*req].pass(sets[*set], "main", shape, Torch::default()) {
                Ok(p) => {
                    passes.insert(req.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {e}"),
            },
            ["endpass", req, peak] => {
                if let Some(p) = passes.remove(*req) {
                    p.end(Measured { allocated_peak: peak.parse().unwrap(), slack: 0, sites: vec![] }, Torch::default());
                }
                "ok".into()
            }
            ["bind", set, block] => match m().bind(sets[*set], block.parse().unwrap(), 0) {
                Ok(pin) => {
                    let fresh = pin.fresh;
                    pins.push(pin);
                    format!("ok {fresh}")
                }
                Err(e) => format!("err {e} sticky={}", e.is_sticky()),
            },
            ["exact", set, block, seed] => {
                let b: u32 = block.parse().unwrap();
                let pin = m().bind(sets[*set], b, 0).unwrap();
                let ok = exact(&pin, &bytes_of(seed, b), 0);
                pins.push(pin);
                format!("ok {ok}")
            }
            ["filled"] => {
                pins.iter().filter(|p| p.fresh).for_each(Pin::filled);
                "ok".into()
            }
            ["stray"] => {
                let code = stray_write(pins.last().unwrap().ptr);
                let e: Error = cuda::check("the stray write", code).map_err(Error::from).err().unwrap();
                format!("ok {code} sticky={}", e.is_sticky())
            }
            ["trace", req] => {
                let origins: Vec<String> = requests[*req].stats().trace.iter().map(|t| format!("{:?}", t.origin)).collect();
                format!("ok {}", origins.join(","))
            }
            ["unpin"] => {
                pins.clear();
                "ok".into()
            }
            ["end", req] => {
                passes.remove(*req);
                let s = requests.remove(*req).unwrap().end();
                format!("ok leftover={} kept={} tier={} store={}", s.leftover, s.moved.kept.bytes, s.moved.tier.bytes, s.moved.store.bytes)
            }
            // A sticky error ends the process: no destructor runs against its context.
            ["exit"] => std::process::exit(0),
            _ => format!("err unknown {line}"),
        };
        println!("@ {reply}");
        std::io::stdout().flush().unwrap();
    }
}

/// A child running [`role_agent`].
struct Agent {
    child: Child,
    to: std::process::ChildStdin,
    from: BufReader<std::process::ChildStdout>,
}

impl Agent {
    fn spawn() -> Agent {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "role_agent", "--nocapture", "--test-threads=1"])
            .env(ROLE, "agent")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let (to, from) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()));
        Agent { child, to, from }
    }

    /// The reply to `line` (libtest may print its own text before it on the line).
    fn say(&mut self, line: &str) -> String {
        writeln!(self.to, "{line}").unwrap();
        self.to.flush().unwrap();
        loop {
            let mut got = String::new();
            assert!(self.from.read_line(&mut got).unwrap() > 0, "the agent ended at {line:?}");
            if let Some((_, reply)) = got.split_once("@ ") {
                return reply.trim().to_string();
            }
        }
    }

    fn ok(&mut self, line: &str) -> String {
        let reply = self.say(line);
        assert!(reply.starts_with("ok"), "{line} -> {reply}");
        reply.trim_start_matches("ok").trim().to_string()
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn exit(mut self) {
        writeln!(self.to, "exit").unwrap();
        assert!(self.child.wait().unwrap().success());
    }
}

/// The case's ledger in this process (TensorD's part: the table and custody), under a name of the
/// case's own; no CUDA context stays in this process.
fn tensord(case: &str) -> (Ledger, GpuId, String) {
    let (b, g, uuid) = card(case);
    step!();
    drop(b);
    step!();
    (Ledger::create(&g, None).unwrap(), g, uuid)
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_kept_set_is_imported_after_its_holders_kill_9_and_another_fill_refills() {
    let (ledger, g, uuid) = tensord("kill");
    step!();
    let mut a = Agent::spawn();
    step!();
    a.ok(&format!("open {}", g.uuid));
    step!();
    a.ok("load k 4 10 stored gpu");
    step!();
    a.child.kill().unwrap();
    step!();
    assert_eq!(a.child.wait().unwrap().signal(), Some(libc::SIGKILL));
    step!();
    ledger.reap(a.pid());
    step!();
    let orphans: u64 = ledger.status().rows.iter().map(|r| r.classes[1]).sum();
    step!();
    assert_eq!(orphans, 4 * REGION_ALIGN, "class O shows the orphans");
    step!();
    let mut s = Agent::spawn();
    step!();
    s.ok(&format!("open {}", g.uuid));
    step!();
    // One load path: the load imports what custody keeps, at any level.
    s.ok("load k 4 10 stored ram");
    step!();
    s.ok("request r k");
    step!();
    s.ok("pass r k 1");
    step!();
    for b in 0..4 {
        assert_eq!(s.ok(&format!("exact k {b} 10")), "true", "block {b} bit-exact");
    }
    assert_eq!(s.ok("trace r"), "Hit,Hit,Hit,Hit", "imported at load, not copied");
    step!();
    s.ok("unpin");
    step!();
    let end = s.ok("end r");
    step!();
    assert!(end.contains("tier=0 store=0"), "{end}");
    step!();
    s.exit();
    let mut c = Agent::spawn();
    step!();
    c.ok(&format!("open {}", g.uuid));
    step!();
    c.ok("load k 4 10 lora-x ram");
    step!();
    c.ok("request r k");
    step!();
    c.ok("pass r k 1");
    step!();
    assert_eq!(c.ok("bind k 0"), "true", "another fill identity is fresh: the caller bakes it");
    step!();
    c.ok("filled");
    step!();
    let trace = c.ok("trace r");
    step!();
    assert!(!trace.contains("Kept"), "{trace}");
    step!();
    c.ok("unpin");
    step!();
    c.ok("end r");
    step!();
    c.exit();
    println!("PASS a kept set is imported after its holder's kill -9, and another fill refills {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn kept_pages_survive_a_holder_that_served() {
    let (ledger, g, uuid) = tensord("served");
    step!();
    let mut a = Agent::spawn();
    step!();
    a.ok(&format!("open {}", g.uuid));
    step!();
    a.ok("load s 3 14 stored gpu");
    step!();
    a.ok("request r s");
    step!();
    a.ok("pass r s 1");
    step!();
    for b in 0..3 {
        a.ok(&format!("bind s {b}"));
    }
    a.ok("unpin");
    step!();
    // A measured pass: the successor's pass needs room, and must not find it in its own kept pages.
    a.ok(&format!("endpass r {}", 1u64 << 30));
    step!();
    a.ok("end r");
    step!();
    a.child.kill().unwrap();
    step!();
    a.child.wait().unwrap();
    step!();
    ledger.reap(a.pid());
    step!();
    let mut b = Agent::spawn();
    step!();
    b.ok(&format!("open {}", g.uuid));
    step!();
    b.ok("load s 3 14 stored ram");
    step!();
    b.ok("request r s");
    step!();
    b.ok("pass r s 1");
    step!();
    for block in 0..3 {
        assert_eq!(b.ok(&format!("exact s {block} 14")), "true");
    }
    assert_eq!(b.ok("trace r"), "Hit,Hit,Hit", "every page kept, none copied");
    step!();
    b.ok("unpin");
    step!();
    let end = b.ok("end r");
    step!();
    assert!(end.contains("tier=0 store=0"), "{end}");
    step!();
    b.exit();
    println!("PASS kept pages survive a holder that served {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn binds_outlive_the_custody_holder() {
    // No TensorD: the first participant serves custody, and it dies.
    let (b, g, uuid) = card("holder");
    step!();
    drop(b);
    step!();
    let mut a = Agent::spawn();
    step!();
    a.ok(&format!("open {}", g.uuid));
    step!();
    a.ok("load hx 2 15 stored gpu");
    step!();
    let mut c = Agent::spawn();
    step!();
    c.ok(&format!("open {}", g.uuid));
    step!();
    c.ok("load hy 2 16 stored ram");
    step!();
    a.child.kill().unwrap();
    step!();
    a.child.wait().unwrap();
    step!();
    c.ok("request r hy");
    step!();
    c.ok("pass r hy 1");
    step!();
    assert_eq!(c.ok("exact hy 1 16"), "true", "binds go on: the next call serves custody itself");
    step!();
    c.ok("unpin");
    step!();
    c.ok("end r");
    step!();
    c.exit();
    // The table goes with the last of its rows.
    drop(Ledger::create(&g, None).unwrap());
    step!();
    println!("PASS binds outlive the custody holder {uuid}");
    step!();
}

/// Under a pressure event, a RAM-tier layout an executor maps and has page-locked with the card
/// (`register_host`) is asked back first: the executor unregisters and unmaps it, then the tier
/// punches it out, so its bytes really return. The executor goes on from the store.
#[test]
#[ignore = "native GPU: a rental only"]
fn the_tier_takes_back_a_pinned_layout() {
    use std::os::fd::FromRawFd;
    let (ledger, g, uuid) = tensord("pinned");
    step!();
    // SAFETY: eventfd(2) with no shared state; the descriptor is ours.
    let event = unsafe { platform::handles::Owned::from_raw_fd(libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)) };
    step!();
    let signal = event.try_clone().unwrap();
    step!();
    let tier = varena::Tier::serve(varena::Pressure::Events(event)).unwrap().expect("no other participant serves this user's RAM tier");
    step!();
    let digest = |n: &str| format!("{n}-2-{}", g.uuid);
    step!();
    let resident = |d: &str| tier.resident().into_iter().find(|(k, _)| *k == d).map_or(0, |(_, b)| b);
    step!();
    let mut y = Agent::spawn();
    step!();
    y.ok(&format!("open {}", g.uuid));
    step!();
    y.ok("load p1 2 17 stored ram");
    step!();
    y.ok("load p2 2 18 stored ram");
    step!();
    // A device page filled from the tier: the region is mapped and page-locked in y.
    assert_eq!(y.ok("exact p1 0 17"), "true");
    step!();
    assert!(resident(&digest("p1")) >= 2 * REGION_ALIGN && resident(&digest("p2")) >= 2 * REGION_ALIGN);
    step!();
    // SAFETY: an 8-byte write to our eventfd.
    unsafe { libc::write(signal.as_raw_fd(), 1u64.to_ne_bytes().as_ptr().cast(), 8) };
    step!();
    tier.await_releases(1);
    step!();
    assert!(resident(&digest("p1")) < REGION_ALIGN, "the pinned layout's regions returned: {}", resident(&digest("p1")));
    step!();
    assert!(resident(&digest("p2")) >= 2 * REGION_ALIGN, "the next one is untouched");
    step!();
    assert!(tier.acked() >= 2 * REGION_ALIGN, "y unregistered and unmapped it when asked: {}", tier.acked());
    step!();
    assert_eq!(y.ok("exact p1 1 17"), "true", "y goes on from the store");
    step!();
    y.ok("unpin");
    step!();
    y.exit();
    drop(ledger);
    step!();
    println!("PASS the tier takes back a pinned layout ({} bytes acked) {uuid}", tier.acked());
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn two_processes_contend_and_an_idle_holder_gives_back_while_its_pins_stay() {
    let (ledger, g, uuid) = tensord("contend");
    step!();
    let held = |pid: u32| ledger.status().rows.iter().find(|r| r.pid == pid).map(|r| r.classes.iter().sum::<u64>()).unwrap();
    step!();
    // Both contexts first: then the holder's idle set takes the rest of the card.
    let mut demander = Agent::spawn();
    step!();
    demander.ok(&format!("open {}", g.uuid));
    step!();
    let mut holder = Agent::spawn();
    step!();
    holder.ok(&format!("open {}", g.uuid));
    step!();
    holder.ok("load h 1 11 stored ram");
    step!();
    holder.ok("load idle more bulk stored gpu");
    step!();
    holder.ok("request rh h");
    step!();
    holder.ok("pass rh h 1");
    step!();
    holder.ok("bind h 0");
    step!();
    let before = held(holder.pid());
    step!();
    demander.ok("load d 4 12 stored ram");
    step!();
    demander.ok("request r d");
    step!();
    demander.ok("pass r d 1");
    step!();
    for b in 0..4 {
        assert_eq!(demander.ok(&format!("exact d {b} 12")), "true", "the demander binds block {b}");
    }
    // The holder's set took all but less than a page of the card: it gave back the rest.
    let given = before.saturating_sub(held(holder.pid()));
    step!();
    assert!(given >= 3 * REGION_ALIGN, "the idle holder gave back {given} of the demander's {}", 4 * REGION_ALIGN);
    step!();
    assert_eq!(holder.ok("exact h 0 11"), "true", "the holder's live pin stayed");
    step!();
    demander.ok("unpin");
    step!();
    demander.ok("end r");
    step!();
    holder.ok("unpin");
    step!();
    holder.ok("end rh");
    step!();
    demander.exit();
    holder.exit();
    println!("PASS two processes contend; an idle holder gives back ({given} bytes) while its pins stay {uuid}");
    step!();
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_stray_write_to_a_kept_page_ends_only_its_writer() {
    let (ledger, g, uuid) = tensord("sticky");
    step!();
    let mut w = Agent::spawn();
    step!();
    w.ok(&format!("open {}", g.uuid));
    step!();
    w.ok("load s 2 13 stored gpu");
    step!();
    w.ok("request r s");
    step!();
    w.ok("pass r s 1");
    step!();
    w.ok("bind s 0");
    step!();
    let fault = w.ok("stray");
    step!();
    assert!(fault.ends_with("sticky=true"), "a write to a kept, read-only page faults sticky: {fault}");
    step!();
    let pid = w.pid();
    step!();
    w.exit();
    ledger.reap(pid);
    step!();
    let mut next = Agent::spawn();
    step!();
    next.ok(&format!("open {}", g.uuid));
    step!();
    next.ok("load s 2 13 stored ram");
    step!();
    next.ok("request r s");
    step!();
    next.ok("pass r s 1");
    step!();
    for b in 0..2 {
        assert_eq!(next.ok(&format!("exact s {b} 13")), "true", "block {b} is the stored bytes: the write never landed");
    }
    let trace = next.ok("trace r");
    step!();
    assert_eq!(trace, "Hit,Hit", "both pages were imported at load");
    step!();
    next.ok("unpin");
    step!();
    let end = next.ok("end r");
    step!();
    assert!(end.contains("tier=0 store=0"), "the kept pages came back, none copied: {end}");
    step!();
    next.exit();
    println!("PASS a stray write to a kept page ends only its writer (fault {fault}; next process {trace}) {uuid}");
    step!();
}
