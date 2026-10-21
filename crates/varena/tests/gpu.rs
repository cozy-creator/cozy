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
use std::sync::mpsc;

use tensorfs::io::{self, cuda, Copy, Dst, ReadOpts, Src};
use varena::{Access, Backend, CudaBackend, Error, Event, Physical, Va};

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
fn context_is_measured_and_free_is_the_drivers() {
    let (b, uuid) = gpu();
    let context = b.context();
    assert!(context > 0 && context < b.capacity(), "the context made at open measures {context}");
    let len = 64 * b.granularity();
    let before = b.free().unwrap();
    let page = b.create(len).unwrap();
    let during = b.free().unwrap();
    b.destroy(page);
    let after = b.free().unwrap();
    assert!(before - during >= len, "creating {len} bytes freed only {}", before - during);
    assert!(after - during >= len, "destroying {len} bytes gave back only {}", after - during);
    println!("PASS context {context} measured, free is the driver's {uuid}");
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

/// One `st.global.u8` of 7 at its argument: a stray write.
const STRAY_WRITE: &std::ffi::CStr = c"
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
";

/// Launches the stray write at `ptr` and returns the driver's verdict on it.
fn stray_write(ptr: u64) -> cuda::CUresult {
    let d = cuda::driver().unwrap();
    let (_, ctx) = cuda::primary_context(0).unwrap();
    let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
    let (mut module, mut kernel) = (std::ptr::null_mut(), std::ptr::null_mut());
    cuda::check("cuModuleLoadData", unsafe { (d.module_load_data)(&mut module, STRAY_WRITE.as_ptr().cast()) }).unwrap();
    cuda::check("cuModuleGetFunction", unsafe { (d.module_get_function)(&mut kernel, module, c"stray".as_ptr()) }).unwrap();
    let mut at = ptr;
    let mut args = [(&mut at as *mut u64).cast::<c_void>()];
    let stream = std::ptr::null_mut();
    match unsafe { (d.launch_kernel)(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args.as_mut_ptr(), std::ptr::null_mut()) } {
        cuda::SUCCESS => unsafe { (d.ctx_synchronize)() },
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

/// Device memory taken with `cuMemAlloc`, as torch takes it.
struct Pressure {
    ctx: cuda::CUcontext,
    taken: Vec<u64>,
}

impl Pressure {
    /// Takes what the device has free, in ever smaller pieces, down to `piece`.
    fn fill(b: &CudaBackend, piece: u64) -> Pressure {
        let d = cuda::driver().unwrap();
        let (_, ctx) = cuda::primary_context(0).unwrap();
        let _ctx = cuda::CtxGuard::enter(ctx).unwrap();
        let mut pressure = Pressure { ctx, taken: Vec::new() };
        let mut size = b.free().unwrap();
        while size >= piece {
            let mut ptr = 0;
            match unsafe { (d.mem_alloc)(&mut ptr, size as usize) } {
                cuda::SUCCESS => pressure.taken.push(ptr),
                cuda::ERROR_OUT_OF_MEMORY => size /= 2,
                code => panic!("cuMemAlloc: {code}"),
            }
        }
        pressure
    }
}

impl Drop for Pressure {
    fn drop(&mut self) {
        let _ctx = cuda::CtxGuard::enter(self.ctx).unwrap();
        let d = cuda::driver().unwrap();
        for ptr in self.taken.drain(..) {
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
    let pressure = Pressure::fill(&b, b.granularity());
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
