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

/// The one GPU the run was given, and its UUID.
fn gpu() -> (CudaBackend, String) {
    let b = CudaBackend::open(0).expect("a CUDA device");
    let mut count = 0;
    cuda::check("cuDeviceGetCount", unsafe { (cuda::driver().unwrap().device_get_count)(&mut count) }).unwrap();
    assert_eq!(count, 1, "the run must see exactly one GPU");
    let uuid = b.gpu().unwrap().uuid;
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
    b.map(va, offset, &page, Access::ReadWrite).unwrap();
    // SAFETY: the mapping is ours, writable, and nothing else touches it during the read.
    unsafe { io::read(&[Copy { src, dst: b.target(va, offset, len) }], &ReadOpts::default()) }.unwrap();
    let ready = b.fence().unwrap();
    b.protect(va, offset, len, Access::Read).unwrap();
    (page, ready)
}

/// The device bytes at `ptr`, read after `stream`'s work.
fn bytes(ptr: u64, len: u64, stream: usize) -> Vec<u8> {
    let mut out = vec![0; len as usize];
    // SAFETY: `ptr` is a live mapping for the whole read; `out` is ours.
    unsafe { io::read(&[Copy { src: Src::Device { device: 0, ptr, stream }, dst: Dst::host(&mut out) }], &ReadOpts::default()) }.unwrap();
    out
}

fn unload(b: &CudaBackend, va: Va, offset: u64, page: Physical) {
    b.unmap(va, offset, page.len).unwrap();
    b.destroy(page);
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
    platform::handles::send_fd(control, handle).unwrap();
}

fn receive(control: &mut UnixStream) -> (OwnedFd, u64) {
    let mut len = [0; 8];
    control.read_exact(&mut len).unwrap();
    (platform::handles::recv_fd(control).unwrap(), u64::from_le_bytes(len))
}

/// The first line of `child`'s stdout after `marker`.
fn marked(child: &mut Child, marker: &str) -> String {
    let lines = BufReader::new(child.stdout.take().unwrap()).lines();
    lines.map_while(Result::ok).find_map(|l| l.split_once(marker).map(|(_, rest)| rest.to_owned())).unwrap()
}

#[test]
#[ignore = "native GPU: a rental only"]
fn free_is_the_drivers() {
    let (b, uuid) = gpu();
    let len = 64 * b.granularity();
    let before = b.free().unwrap();
    let page = b.create(len).unwrap();
    let during = b.free().unwrap();
    b.destroy(page);
    let after = b.free().unwrap();
    assert!(before - during >= len, "creating {len} bytes took only {}", before - during);
    assert!(after - during >= len, "destroying {len} bytes gave back only {}", after - during);
    println!("PASS free is the driver's {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn process_used_counts_this_processes_pages_and_allocations() {
    let (b, uuid) = gpu();
    let reading = platform::gpu::Device::open(&uuid).unwrap().read().unwrap();
    let how = match reading.processes.is_some_and(|p| p.contains_key(&std::process::id())) {
        true => "NVML per process",
        false => "device delta since open",
    };
    let len = 64 * b.granularity();
    let start = b.process_used().unwrap();
    assert!(start > 0, "this process holds a context");
    let context = start;
    let va = b.reserve(len).unwrap();
    let page = b.create(len).unwrap();
    b.map(va, 0, &page, Access::ReadWrite).unwrap();
    let paged = b.process_used().unwrap();
    assert!(paged - start >= len, "a {len} byte page moved process bytes by {}", paged - start);
    let pressure = Pressure::take(len);
    let allocated = b.process_used().unwrap();
    assert!(allocated - paged >= len, "a {len} byte allocation moved process bytes by {}", allocated - paged);
    drop(pressure);
    unload(&b, va, 0, page);
    b.release(va);
    let (mut control, theirs) = UnixStream::pair().unwrap();
    let mut holder = child("role_holder", "holder", Stdio::from(OwnedFd::from(theirs)));
    let (kept, len) = receive(&mut control);
    marked(&mut holder, "HOLDER ");
    let before = b.process_used().unwrap();
    let page = b.import(kept, len).unwrap();
    let va = b.reserve(len).unwrap();
    b.map(va, 0, &page, Access::Read).unwrap();
    let imported = b.process_used().unwrap() as i64 - before as i64;
    holder.kill().unwrap();
    holder.wait().unwrap();
    unload(&b, va, 0, page);
    b.release(va);
    println!("PASS process bytes ({how}): context {context} at open, pages and allocations counted; a {len} byte imported page moved them {imported} {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn pages_fill_protect_unmap_and_survive_a_clean_owner_exit() {
    let (b, uuid) = gpu();
    let len = 2 * b.granularity();
    let (first, second) = (pattern(len, 1), pattern(len, 2));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("first");
    std::fs::write(&path, &first).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let va = b.reserve(2 * len).unwrap();
    let (p1, r1) = filled(&b, va, 0, len, Src::File(&file, 0));
    let (p2, r2) = filled(&b, va, len, len, Src::Host(second.as_ptr()));
    let consumer = cuda::Stream::new(0).unwrap();
    b.wait(r1, consumer.raw()).unwrap();
    b.wait(r2, consumer.raw()).unwrap();
    assert_eq!(bytes(va.base, len, consumer.raw()), first);
    assert_eq!(bytes(va.base + len, len, consumer.raw()), second);
    unload(&b, va, len, p2);
    assert_eq!(bytes(va.base, len, consumer.raw()), first, "unmapping one page leaves its neighbour");
    let release = b.record(consumer.raw()).unwrap();
    consumer.drain().unwrap();
    assert!(b.query(release).unwrap());
    let kept = b.export(&p1).unwrap();
    unload(&b, va, 0, p1);
    b.release(va);
    for e in [r1, r2, release] {
        b.retire(e);
    }
    drop((consumer, b));

    let next = CudaBackend::open(0).unwrap();
    let page = next.import(kept, len).unwrap();
    let va = next.reserve(len).unwrap();
    next.map(va, 0, &page, Access::Read).unwrap();
    assert_eq!(bytes(va.base, len, 0), first, "the kept page outlived its owner's exit");
    unload(&next, va, 0, page);
    next.release(va);
    println!("PASS pages fill, protect, unmap and survive a clean owner exit {uuid}");
}

/// `stray`: one `st.global.u8` of 7 at its argument. `copy`: `words` u64s from `src` to `dst`,
/// one thread (a consumer that reads a page).
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
    let (_, ctx) = cuda::primary_context(0).unwrap();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let (mut module, mut kernel) = (std::ptr::null_mut(), std::ptr::null_mut());
    cuda::check("cuModuleLoadData", unsafe { (d.module_load_data)(&mut module, KERNELS.as_ptr().cast()) }).unwrap();
    cuda::check("cuModuleGetFunction", unsafe { (d.module_get_function)(&mut kernel, module, name.as_ptr()) }).unwrap();
    let mut values = args.to_vec();
    let mut params: Vec<*mut c_void> = values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    unsafe { (d.launch_kernel)(kernel, 1, 1, 1, 1, 1, 1, 0, stream as cuda::CUstream, params.as_mut_ptr(), std::ptr::null_mut()) }
}

/// The context's verdict on everything queued so far.
fn synchronize() -> cuda::CUresult {
    let (_, ctx) = cuda::primary_context(0).unwrap();
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
    let (handle, len) = receive(&mut parent());
    let page = b.import(handle, len).unwrap();
    let va = b.reserve(len).unwrap();
    b.map(va, 0, &page, Access::Read).unwrap();
    let code = stray_write(va.base);
    let e: Error = cuda::check("the stray write", code).expect_err("a write to a read-only page must fault").into();
    assert!(e.is_sticky(), "a read-only fault is sticky: {e}");
    println!("FAULT {code} {uuid}");
    std::io::stdout().flush().unwrap();
    // The context is gone: exit without running destructors against it.
    std::process::exit(0);
}

#[test]
#[ignore = "native GPU: a rental only"]
fn read_only_page_faults_a_stray_write_only_in_the_writers_process() {
    let (b, uuid) = gpu();
    let len = b.granularity();
    let data = pattern(len, 3);
    let va = b.reserve(len).unwrap();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    let (mut control, theirs) = UnixStream::pair().unwrap();
    let mut writer = child("role_stray_writer", "writer", Stdio::from(OwnedFd::from(theirs)));
    send(&mut control, &b.export(&page).unwrap(), len);
    let fault = marked(&mut writer, "FAULT ");
    assert!(writer.wait().unwrap().success());
    let (code, theirs) = fault.split_once(' ').unwrap();
    assert_eq!(theirs, uuid, "the writer faulted on another GPU");
    b.wait(ready, 0).unwrap();
    assert_eq!(bytes(va.base, len, 0), data, "the stray write must not land");
    let other = b.reserve(len).unwrap();
    let (fresh, done) = filled(&b, other, 0, len, Src::Host(data.as_ptr()));
    b.wait(done, 0).unwrap();
    assert_eq!(bytes(other.base, len, 0), data, "the holder serves on after the writer's fault");
    unload(&b, other, 0, fresh);
    unload(&b, va, 0, page);
    b.release(other);
    b.release(va);
    println!("PASS a stray write faults only its writer (CUresult {code}) {uuid}");
}

/// The device's free bytes, as the driver measures them.
fn driver_free() -> u64 {
    let (_, ctx) = cuda::primary_context(0).unwrap();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let (mut free, mut total) = (0usize, 0usize);
    cuda::check("cuMemGetInfo", unsafe { (cuda::driver().unwrap().mem_get_info)(&mut free, &mut total) }).unwrap();
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
    let len = 4 * b.granularity();
    let data = pattern(len, 4);
    let pressure = Pressure::fill(b.granularity());
    let left = b.free().unwrap();
    let e = b.create(len).expect_err("a full device refuses a page");
    assert!(matches!(e, Error::Device { code: 2, .. }), "expected CUDA_ERROR_OUT_OF_MEMORY, got {e}");
    assert!(!e.is_sticky());
    drop(pressure);
    let va = b.reserve(len).unwrap();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    b.wait(ready, 0).unwrap();
    assert_eq!(bytes(va.base, len, 0), data, "the same process fills after its OOM");
    unload(&b, va, 0, page);
    b.release(va);
    println!("PASS real CUDA OOM recovered in process (free {left} at the OOM) {uuid}");
}

#[test]
#[ignore = "native GPU: a role of kept_page_is_imported_bit_exact_after_its_holders_kill_9"]
fn role_holder() {
    if role().as_deref() != Some("holder") {
        return;
    }
    let (b, uuid) = gpu();
    let len = 4 * b.granularity();
    let data = pattern(len, 5);
    let va = b.reserve(len).unwrap();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    b.wait(ready, 0).unwrap();
    let mut control = parent();
    send(&mut control, &b.export(&page).unwrap(), len);
    println!("HOLDER {uuid}");
    std::io::stdout().flush().unwrap();
    // Held until killed.
    control.read_exact(&mut [0]).unwrap();
    unreachable!("the holder is killed, never let go");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn kept_page_is_imported_bit_exact_after_its_holders_kill_9() {
    let (mut control, theirs) = UnixStream::pair().unwrap();
    let mut holder = child("role_holder", "holder", Stdio::from(OwnedFd::from(theirs)));
    let (kept, len) = receive(&mut control);
    let uuid = marked(&mut holder, "HOLDER ");
    holder.kill().unwrap();
    assert_eq!(holder.wait().unwrap().signal(), Some(libc::SIGKILL));
    let (b, mine) = gpu();
    assert_eq!(mine, uuid);
    let page = b.import(kept, len).unwrap();
    let va = b.reserve(len).unwrap();
    b.map(va, 0, &page, Access::Read).unwrap();
    assert_eq!(bytes(va.base, len, 0), pattern(len, 5), "the kept page is the holder's, bit-exact");
    let before = b.free().unwrap();
    unload(&b, va, 0, page);
    b.release(va);
    let after = b.free().unwrap();
    assert!(after - before >= len, "the last reference gone, its {len} bytes are free (got {})", after - before);
    println!("PASS a kept page is imported bit-exact after its holder's kill -9 {uuid}");
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
    let len = b.granularity();
    let data = pattern(len, 6);
    let va = b.reserve(len).unwrap();
    let (page, ready) = filled(&b, va, 0, len, Src::Host(data.as_ptr()));
    let consumer = cuda::Stream::new(0).unwrap();
    let other = cuda::Stream::new(0).unwrap();
    b.wait(ready, consumer.raw()).unwrap();
    let gate = Gate::hold(consumer.raw());
    let release = b.record(consumer.raw()).unwrap();
    b.wait(release, other.raw()).unwrap();
    let after = b.record(other.raw()).unwrap();
    assert!(!b.query(release).unwrap(), "a pin's release is pending while its consumer runs");
    assert!(!b.query(after).unwrap(), "a stream waiting on it waits in-stream");
    assert_eq!(b.elapsed(release, after).unwrap(), None);
    drop(gate);
    other.drain().unwrap();
    assert!(b.query(release).unwrap() && b.query(after).unwrap());
    assert!(b.elapsed(release, after).unwrap().is_some());
    assert_eq!(bytes(va.base, len, other.raw()), data);
    for e in [ready, release, after] {
        b.retire(e);
    }
    unload(&b, va, 0, page);
    b.release(va);
    b.drain().unwrap();
    println!("PASS release events order consumers without a host wait {uuid}");
}

/// Drops `path`'s pages from the page cache, so the next read comes from the disk.
fn uncached(path: &std::path::Path) {
    let file = std::fs::File::open(path).unwrap();
    file.sync_all().unwrap();
    assert_eq!(unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) }, 0);
}

/// The GB/s of the copy runs TensorFS logged since the last call.
fn rate() -> f64 {
    let runs: Vec<_> = io::metrics::take().into_iter().filter(|c| c.kind == "h2d").collect();
    let (bytes, ns) = runs.iter().fold((0, 0), |(b, n), c| (b + c.bytes, n + c.ns));
    bytes as f64 / ns.max(1) as f64
}

#[test]
#[ignore = "native GPU: a rental only"]
fn fills_are_reported_against_the_measured_ceiling() {
    let (b, uuid) = gpu();
    // A gibibyte: enough for a fill to stand for the link, as TensorFS's own H2D probe is 128 MiB.
    let len = (1u64 << 30).next_multiple_of(b.granularity());
    let data = pattern(len, 7);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights");
    std::fs::write(&path, &data).unwrap();
    let ceiling = io::ceiling(Some(0), dir.path()).unwrap();
    let h2d = ceiling.h2d_gbps.unwrap();
    let va = b.reserve(len).unwrap();
    let mut report = Vec::new();
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
    println!("FILLS {} (disk {:.2} GB/s direct={})", report.join("; "), ceiling.disk_gbps, ceiling.disk_direct);
    println!("PASS fills reported against the measured ceiling {uuid}");
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
    let (h2d, _) = io::metrics::h2d(0).unwrap();
    let len = (1u64 << 30).next_multiple_of(b.granularity());
    let data = pattern(len, 9);
    let va = b.reserve(len).unwrap();
    let page = b.create(len).unwrap();
    b.map(va, 0, &page, Access::ReadWrite).unwrap();
    let mut rows = vec![];
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
    read("unregistered 1");
    let (again, ..) = read("unregistered 2");
    read("unregistered 3");
    // SAFETY: `data` stays mapped until it is unregistered below.
    unsafe { b.register_host(data.as_ptr(), len) }.unwrap();
    let first = read("registered 1");
    let second = read("registered 2");
    b.unregister_host(data.as_ptr()).unwrap();
    assert_eq!(bytes(va.base, len, 0), data);
    b.unmap(va, 0, len).unwrap();
    b.destroy(page);
    b.release(va);
    println!("REUSE of {h2d:.2} GB/s: {}", rows.join("; "));
    for (gbps, pin, staged) in [first, second] {
        assert_eq!((pin, staged), (0, 0), "a registered range copies directly, never staged or pinned per read");
        assert!(gbps > again, "a registered read beats an unregistered reuse ({gbps:.2} vs {again:.2} GB/s)");
    }
    println!("PASS a registered host range fills directly every time (second {:.0}% vs first {:.0}%) {uuid}", 100.0 * second.0 / h2d, 100.0 * first.0 / h2d);
}

// ---- Memory over the CUDA backend

/// The run's GPU under a name of this case's own, so its table, custody and kept pages are the
/// case's (a machine's TensorD may run beside it); and the real UUID.
fn card(case: &str) -> (CudaBackend, GpuId, String) {
    let (b, uuid) = gpu();
    let g = GpuId { uuid: format!("{uuid}-{case}-{}", std::process::id()), ..b.gpu().unwrap() };
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
    let fingerprint = Fingerprint::new(&layout, &[], fill, "release-1", "0/1", uuid);
    SetSpec { name: name.into(), layout: Arc::new(layout), source: Arc::new(Source::files(vec![], false, 0)), fingerprint }
}

/// The bytes every page of a [`bulk`] set holds.
fn bulk_block() -> Arc<[u8]> {
    block(0xb, 0)
}

/// A set of `blocks` pages that all hold one page of bytes: a large set costs one page of RAM.
fn bulk(name: &str, blocks: u32, uuid: &str) -> SetSpec {
    let one = bulk_block();
    set(name, blocks, Fill::Stored, uuid, |_| one.clone())
}

/// Bytes the cases leave free for pages (a filler set of more pages than fit takes them all); the
/// rest of the card goes to allocations no one can free. Two gibibytes: room for a one-gibibyte
/// activation to be refused and then made.
const ROOM: u64 = 2 << 30;

/// Takes the card down to [`ROOM`] free, as other allocations of this process would.
fn crowd() -> Pressure {
    let mut p = Pressure::none();
    let over = driver_free().saturating_sub(ROOM);
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
fn activation(r: &Request, p: &mut Pressure, len: u64, pages: u32) -> Vec<String> {
    let mut answers = vec![];
    while let Err(code) = p.try_take(len) {
        assert_eq!(code, cuda::ERROR_OUT_OF_MEMORY);
        let next = r.oom(Some(len), None, p.torch());
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
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let s = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(1, b)), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    let pass = r.pass(s, "1x1", Torch::default()).unwrap();
    let streams = [cuda::Stream::new(0).unwrap(), cuda::Stream::new(0).unwrap()];
    let pins: Vec<Pin> = std::thread::scope(|t| {
        let binds: Vec<_> = streams.iter().map(|st| t.spawn(|| mem.bind(s, 0, st.raw()).unwrap())).collect();
        binds.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (pin, st) in pins.iter().zip(&streams) {
        assert!(exact(pin, &block(1, 0), st.raw()));
    }
    assert_eq!(pins[0].ptr, pins[1].ptr, "one home");
    let moved = r.stats().moved;
    assert_eq!(moved.store.bytes + moved.tier.bytes, REGION_ALIGN, "one fill for both binders");
    drop(pins);
    pass.end(Measured::default(), Torch::default());
    assert_eq!(r.end().leftover, 0);
    println!("PASS concurrent binds of one block fill it once {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_page_unmaps_only_after_its_last_consumer() {
    let (b, g, uuid) = card("release");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    // A declared fill: the bind is fresh, and a pin dropped untransformed sends its page away.
    let s = mem.load(set("t", 1, Fill::Declared("bake-1".into()), &g.uuid, |b| block(2, b)), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["t"]])).unwrap();
    let pass = r.pass(s, "1x1", Torch::default()).unwrap();
    let consumer = cuda::Stream::new(0).unwrap();
    let gate = Gate::hold(consumer.raw());
    let pin = mem.bind(s, 0, consumer.raw()).unwrap();
    assert!(pin.fresh);
    let scratch = Pressure::take(REGION_ALIGN);
    let (dst, _) = scratch.taken[0];
    assert_eq!(launch(c"copy", &[dst, pin.ptr, REGION_ALIGN / 8], consumer.raw()), cuda::SUCCESS);
    drop(pin);
    // Memory takes the release now; its event waits behind the gate and the copy. A second bind
    // must wait for it too.
    let _ = mem.torch_limit();
    let again = std::thread::scope(|t| {
        let rebind = t.spawn(|| mem.bind(s, 0, 0));
        drop(gate);
        rebind.join().unwrap()
    });
    assert_eq!(synchronize(), cuda::SUCCESS, "the consumer read its page while it was mapped");
    assert_eq!(bytes(dst, REGION_ALIGN, 0), *block(2, 0));
    let again = again.unwrap();
    assert!(again.fresh, "an untransformed page is filled again");
    assert!(exact(&again, &block(2, 0), 0));
    again.filled();
    drop(again);
    pass.end(Measured::default(), Torch::default());
    assert_eq!(r.end().leftover, 0);
    println!("PASS a page unmaps only after its last consumer {uuid}");
}

/// A pass of `m` (two pinned blocks) whose one-gibibyte activation meets a real OOM: the card is
/// crowded and the rest full of an idle set. Returns the OOM answers and the record.
fn first_run(mem: &Memory, m: SetId, idle: u32) -> (Vec<String>, Stats) {
    let r = mem.request("first", plan(&[&["m"]])).unwrap();
    let pass = r.pass(m, "1x8", Torch::default()).unwrap();
    let pins: Vec<Pin> = (0..2).map(|b| mem.bind(m, b, 0).unwrap()).collect();
    let mut p = Pressure::none();
    let answers = activation(&r, &mut p, ROOM / 2, idle);
    for (b, pin) in pins.iter().enumerate() {
        assert!(exact(pin, &block(3, b as u32), 0), "pins outlive the OOM");
    }
    pass.end(Measured { allocated_peak: ROOM / 2, slack: 0, sites: vec![] }, p.torch());
    drop((pins, p));
    (answers, r.end())
}

#[test]
#[ignore = "native GPU: a rental only"]
fn an_oom_in_a_pass_is_recovered_in_process() {
    let (b, g, uuid) = card("oom");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(3, b)), Level::Ram).unwrap();
    let _crowd = crowd();
    let idle = more_than_free();
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    let (answers, stats) = first_run(&mem, m, idle);
    assert!(!answers.is_empty(), "the full card refused the activation");
    assert!(stats.retries >= 1 && stats.evicted.iter().sum::<u64>() >= ROOM / 2, "{stats:?}");
    assert_eq!(stats.leftover, 0);
    println!("PASS an OOM in a pass is recovered in process (answers {answers:?}) {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_measured_pass_gets_its_room_before_it_runs() {
    let (b, g, uuid) = card("room");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(3, b)), Level::Ram).unwrap();
    let _crowd = crowd();
    let idle = more_than_free();
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    let (answers, _) = first_run(&mem, m, idle);
    assert!(!answers.is_empty(), "the first run met a real OOM");
    // The idle set takes the card again; the second run's pass makes its measured room first.
    mem.load(bulk("idle", idle, &g.uuid), Level::Gpu).unwrap();
    let r = mem.request("second", plan(&[&["m"]])).unwrap();
    let pass = r.pass(m, "1x8", Torch::default()).unwrap();
    let mut p = Pressure::none();
    assert_eq!(p.try_take(ROOM / 2).map(|_| ()), Ok(()), "the pass made its measured room before it ran");
    pass.end(Measured { allocated_peak: ROOM / 2, slack: 0, sites: vec![] }, p.torch());
    let stats = r.end();
    assert_eq!(stats.retries, 0, "{stats:?}");
    println!("PASS a measured pass gets its room before it runs {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_page_larger_than_the_card_is_exhausted_before_any_io_and_its_id_serves_again() {
    let (b, g, uuid) = card("huge");
    let _ledger = Ledger::create(&g, None).unwrap();
    let span = (b.capacity() + REGION_ALIGN).next_multiple_of(REGION_ALIGN);
    let mem = memory(b, &g);
    let one: Arc<[u8]> = vec![7u8].into();
    let items = vec![Item { offset: 0, len: 1, source: ItemSource::Inline(one), part_end: true }];
    let layout = Layout { parts: vec![], regions: vec![Region { offset: 0, nbytes: span, span, items }], nbytes: span, digest: format!("huge-{}", g.uuid), names: vec![] };
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", &g.uuid);
    let source = Arc::new(Source::files(vec![], false, 0));
    // To the GPU: it maps only into free bytes, so the page is never filled (the RAM tier would be).
    let huge = mem.load(SetSpec { name: "huge".into(), layout: Arc::new(layout), source, fingerprint }, Level::Gpu).unwrap();
    io::metrics::take();
    let r = mem.request("r", plan(&[&["huge"]])).unwrap();
    let pass = r.pass(huge, "1", Torch::default());
    let e = match &pass {
        Err(e) => e.clone(),
        Ok(_) => mem.bind(huge, 0, 0).err().expect("a page larger than the card cannot bind"),
    };
    assert!(matches!(e, Error::Exhausted(_)), "{e}");
    drop(pass);
    assert!(io::metrics::take().iter().all(|c| c.bytes == 0), "no byte moved before the refusal");
    let stats = r.end();
    assert_eq!((stats.leftover, stats.moved.store.bytes + stats.moved.tier.bytes), (0, 0), "{stats:?}");
    let m = mem.load(set("m", 1, Fill::Stored, &g.uuid, |b| block(4, b)), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    let pass = r.pass(m, "1", Torch::default()).unwrap();
    let pin = mem.bind(m, 0, 0).unwrap();
    assert!(exact(&pin, &block(4, 0), 0), "the same request id serves again");
    drop(pin);
    pass.end(Measured::default(), Torch::default());
    assert_eq!(r.end().leftover, 0);
    println!("PASS a page larger than the card is exhausted before any I/O, and its id serves again {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_bind_on_a_full_card_takes_idle_pages_then_is_exhausted_and_recovers_when_room_returns() {
    let (b, g, uuid) = card("full");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let m = mem.load(set("m", 2, Fill::Stored, &g.uuid, |b| block(5, b)), Level::Ram).unwrap();
    mem.load(set("idle", 1, Fill::Stored, &g.uuid, |b| block(6, b)), Level::Gpu).unwrap();
    let r = mem.request("r", plan(&[&["m"]])).unwrap();
    let pass = r.pass(m, "1", Torch::default()).unwrap();
    // Everything else on the card is an allocation varena cannot free.
    let full = Pressure::fill(64 << 10);
    let first = mem.bind(m, 0, 0).expect("the idle page makes room");
    let e = mem.bind(m, 1, 0).err().expect("nothing is left to free");
    assert!(matches!(e, Error::Exhausted(_)), "{e}");
    drop(full);
    let second = mem.bind(m, 1, 0).expect("room came back: the same request binds");
    assert!(exact(&first, &block(5, 0), 0) && exact(&second, &block(5, 1), 0));
    drop((first, second));
    pass.end(Measured::default(), Torch::default());
    let stats = r.end();
    assert!(stats.evicted.iter().sum::<u64>() >= REGION_ALIGN && stats.leftover == 0, "{stats:?}");
    println!("PASS a bind on a full card takes idle pages, then is exhausted, and recovers when room returns {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn repeated_passes_of_a_set_keep_its_declared_step() {
    let (b, g, uuid) = card("steps");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let names = ["a", "vae", "audio", "dit", "lora"];
    let ids: Vec<SetId> = names.iter().zip(1u8..).map(|(n, s)| mem.load(set(n, 1, Fill::Stored, &g.uuid, |b| block(s, b)), Level::Ram).unwrap()).collect();
    let id = |n: &str| ids[names.iter().position(|x| *x == n).unwrap()];
    let r = mem.request("r", plan(&[&["a"], &["vae", "audio"], &["dit", "lora"], &["audio"], &["vae"]])).unwrap();
    let forward = |n: &str| {
        let pass = r.pass(id(n), "1", Torch::default()).unwrap();
        drop(mem.bind(id(n), 0, 0).unwrap());
        pass.end(Measured::default(), Torch::default());
        r.stats().trace.last().map(|t| t.step).unwrap()
    };
    assert_eq!(forward("a"), 0);
    assert_eq!(forward("audio"), 1);
    for _ in 0..3 {
        assert_eq!(forward("vae"), 1, "a repeated encode keeps its step, not the decode's");
    }
    assert_eq!((forward("dit"), forward("lora")), (2, 2));
    assert_eq!((forward("audio"), forward("vae")), (3, 4));
    assert_eq!(r.end().leftover, 0);
    println!("PASS repeated passes of a set keep its declared step {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_transform_binds_another_set_under_pressure_and_a_second_binder_waits_for_it() {
    let (b, g, uuid) = card("transform");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let base = mem.load(set("base", 1, Fill::Declared("lora-1".into()), &g.uuid, |b| block(7, b)), Level::Ram).unwrap();
    let factors = mem.load(set("factors", 1, Fill::Stored, &g.uuid, |b| block(8, b)), Level::Ram).unwrap();
    let _crowd = crowd();
    mem.load(bulk("idle", more_than_free(), &g.uuid), Level::Gpu).unwrap();
    let r = mem.request("r", plan(&[&["base", "factors"]])).unwrap();
    let pass = r.pass(base, "1", Torch::default()).unwrap();
    let stream = cuda::Stream::new(0).unwrap();
    let pin = mem.bind(base, 0, stream.raw()).unwrap();
    assert!(pin.fresh, "the caller bakes it");
    // The bake binds the factors on the card the idle set fills, and copies them in.
    let f = mem.bind(factors, 0, stream.raw()).unwrap();
    assert_eq!(launch(c"copy", &[pin.ptr, f.ptr, REGION_ALIGN / 8], stream.raw()), cuda::SUCCESS);
    let second = std::thread::scope(|t| {
        let waiter = t.spawn(|| mem.bind(base, 0, 0).unwrap());
        pin.filled();
        waiter.join().unwrap()
    });
    assert!(!second.fresh, "the second binder gets the baked page");
    assert!(exact(&second, &block(8, 0), 0), "never a half-made page");
    drop((pin, f, second));
    pass.end(Measured::default(), Torch::default());
    assert_eq!(r.end().leftover, 0);
    println!("PASS a transform binds another set under pressure, and a second binder waits for it {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_ram_level_set_binds_from_the_tier() {
    let (b, g, uuid) = card("tier");
    let _ledger = Ledger::create(&g, None).unwrap();
    let mem = memory(b, &g);
    let s = mem.load(set("t", 4, Fill::Stored, &g.uuid, |b| block(9, b)), Level::Ram).unwrap();
    let r = mem.request("r", plan(&[&["t"]])).unwrap();
    let pass = r.pass(s, "1", Torch::default()).unwrap();
    let pins: Vec<Pin> = (0..4).map(|b| mem.bind(s, b, 0).unwrap()).collect();
    for (b, pin) in pins.iter().enumerate() {
        assert!(exact(pin, &block(9, b as u32), 0));
    }
    drop(pins);
    pass.end(Measured::default(), Torch::default());
    let stats = r.end();
    let origins: Vec<String> = stats.trace.iter().map(|t| format!("{:?}", t.origin)).collect();
    assert_eq!(stats.moved.tier.bytes, 4 * REGION_ALIGN, "{origins:?}");
    let gbps = stats.moved.tier.bytes as f64 / stats.moved.tier.ns.max(1) as f64;
    println!("FILLS tier {gbps:.2} GB/s ({origins:?})");
    println!("PASS a RAM-level set binds from the tier {uuid}");
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
    let mut uuid = String::new();
    let mut sets: HashMap<String, SetId> = HashMap::new();
    let mut requests: HashMap<String, Request> = HashMap::new();
    let mut passes: HashMap<String, Pass> = HashMap::new();
    let mut pins: Vec<Pin> = vec![];
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
            ["pass", req, set, shape] => match requests[*req].pass(sets[*set], shape, Torch::default()) {
                Ok(p) => {
                    passes.insert(req.to_string(), p);
                    "ok".into()
                }
                Err(e) => format!("err {e}"),
            },
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
    drop(b);
    (Ledger::create(&g, None).unwrap(), g, uuid)
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_kept_set_is_imported_after_its_holders_kill_9_and_another_fill_refills() {
    let (ledger, g, uuid) = tensord("kill");
    let mut a = Agent::spawn();
    a.ok(&format!("open {}", g.uuid));
    a.ok("load k 4 10 stored gpu");
    a.child.kill().unwrap();
    assert_eq!(a.child.wait().unwrap().signal(), Some(libc::SIGKILL));
    ledger.reap(a.pid());
    let mut s = Agent::spawn();
    s.ok(&format!("open {}", g.uuid));
    s.ok("load k 4 10 stored ram");
    s.ok("request r k");
    s.ok("pass r k 1");
    for b in 0..4 {
        assert_eq!(s.ok(&format!("exact k {b} 10")), "true", "block {b} bit-exact");
    }
    assert_eq!(s.ok("trace r"), "Kept,Kept,Kept,Kept", "imported, not copied");
    s.ok("unpin");
    let end = s.ok("end r");
    assert!(end.contains(&format!("kept={}", 4 * REGION_ALIGN)) && end.contains("store=0"), "{end}");
    s.exit();
    let mut c = Agent::spawn();
    c.ok(&format!("open {}", g.uuid));
    c.ok("load k 4 10 lora-x ram");
    c.ok("request r k");
    c.ok("pass r k 1");
    assert_eq!(c.ok("bind k 0"), "true", "another fill identity is fresh: the caller bakes it");
    c.ok("filled");
    let trace = c.ok("trace r");
    assert!(!trace.contains("Kept"), "{trace}");
    c.ok("unpin");
    c.ok("end r");
    c.exit();
    println!("PASS a kept set is imported after its holder's kill -9, and another fill refills {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn two_processes_contend_and_an_idle_holder_gives_back_while_its_pins_stay() {
    let (ledger, g, uuid) = tensord("contend");
    let held = |pid: u32| ledger.status().rows.iter().find(|r| r.pid == pid).map(|r| r.classes.iter().sum::<u64>()).unwrap();
    // Both contexts first: then the holder's idle set takes the rest of the card.
    let mut demander = Agent::spawn();
    demander.ok(&format!("open {}", g.uuid));
    let mut holder = Agent::spawn();
    holder.ok(&format!("open {}", g.uuid));
    holder.ok("load h 1 11 stored ram");
    holder.ok("load idle more bulk stored gpu");
    holder.ok("request rh h");
    holder.ok("pass rh h 1");
    holder.ok("bind h 0");
    let before = held(holder.pid());
    demander.ok("load d 4 12 stored ram");
    demander.ok("request r d");
    demander.ok("pass r d 1");
    for b in 0..4 {
        assert_eq!(demander.ok(&format!("exact d {b} 12")), "true", "the demander binds block {b}");
    }
    let after = held(holder.pid());
    assert!(before - after >= 4 * REGION_ALIGN, "the holder gave back {} of {}", before - after, 4 * REGION_ALIGN);
    assert_eq!(holder.ok("exact h 0 11"), "true", "the holder's live pin stayed");
    demander.ok("unpin");
    demander.ok("end r");
    holder.ok("unpin");
    holder.ok("end rh");
    demander.exit();
    holder.exit();
    println!("PASS two processes contend; an idle holder gives back while its pins stay {uuid}");
}

#[test]
#[ignore = "native GPU: a rental only"]
fn a_stray_write_to_a_kept_page_ends_only_its_writer() {
    let (ledger, g, uuid) = tensord("sticky");
    let mut w = Agent::spawn();
    w.ok(&format!("open {}", g.uuid));
    w.ok("load s 2 13 stored gpu");
    w.ok("request r s");
    w.ok("pass r s 1");
    w.ok("bind s 0");
    let fault = w.ok("stray");
    assert!(fault.ends_with("sticky=true"), "a write to a kept, read-only page faults sticky: {fault}");
    let pid = w.pid();
    w.exit();
    ledger.reap(pid);
    let mut next = Agent::spawn();
    next.ok(&format!("open {}", g.uuid));
    next.ok("load s 2 13 stored ram");
    next.ok("request r s");
    next.ok("pass r s 1");
    for b in 0..2 {
        assert_eq!(next.ok(&format!("exact s {b} 13")), "true", "block {b} is the stored bytes: the write never landed");
    }
    let trace = next.ok("trace r");
    next.ok("unpin");
    next.ok("end r");
    next.exit();
    println!("PASS a stray write to a kept page ends only its writer (fault {fault}; next process {trace}) {uuid}");
}
