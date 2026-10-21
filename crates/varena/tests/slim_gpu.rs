//! Run only on an explicitly rented GPU. This is the real driver/read engine, not a simulator.
#![cfg(target_os = "linux")]
use std::{fs::OpenOptions, io::Write, sync::Arc};
use tensorfs::io::source::{Item, ItemSource, Source};
use tensorfs::{
    format::ObjectRef,
    io::{self, Copy, Dst, ObjectRange, ReadOpts, Src},
};
use varena::gpu::Gpu;

fn bytes(pin: &varena::gpu::Pin) -> Vec<u8> {
    let mut out = vec![0; pin.bytes as usize];
    pin.wait_on(0).expect("consumer stream");
    unsafe {
        io::read(
            &[Copy {
                src: Src::Device {
                    device: 0,
                    ptr: pin.ptr,
                    stream: 0,
                },
                dst: Dst::host(&mut out),
            }],
            &ReadOpts::default(),
        )
    }
    .expect("TensorFS device read");
    out
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn pages_pin_evict_refill_and_survive_clean_owner_exit() {
    let uuid = one_device();
    let data: Vec<u8> = (0..2 * 1024 * 1024).map(|n| (n % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!("varena-slim-gpu-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(&path)
        .expect("source");
    file.write_all(&data).expect("write source");
    let object = ObjectRef::of(&data);
    let source = Arc::new(Source::files(vec![(object.clone(), file)], false, 0));
    let item = Item {
        offset: 0,
        len: data.len() as u64,
        source: ItemSource::Object(ObjectRange {
            obj: object,
            off: 0,
            len: data.len() as u64,
        }),
        part_end: true,
    };
    let gpu = Gpu::open(0).expect("real CUDA device");
    let reserved = gpu
        .reserve(2 * data.len() as u64)
        .expect("stable addresses");
    for (id, offset) in [(1, 0), (2, data.len() as u64)] {
        gpu.register(
            id,
            reserved.clone(),
            offset,
            data.len() as u64,
            source.clone(),
            vec![item.clone()],
            0,
            "same-layout/encoding/transform/rank/device".into(),
        )
        .expect("register page");
        gpu.fill(id).expect("real TensorFS disk fill");
    }
    let pin = gpu.bind(1, 0).expect("bind");
    let _busy_custody = gpu.export(1).expect("read-only custody while pinned");
    assert_eq!(bytes(&pin), data);
    assert!(!gpu.unmap(1).expect("pinned page cannot evict"));
    assert!(gpu.unmap(2).expect("other page independently evicts"));
    let fault = gpu.fill_with(2, |_| panic!("injected page transform panic"));
    assert!(matches!(fault, Err(varena::Error::Internal(_))));
    gpu.recover().expect("failed request recovers in process");
    gpu.fill_from_ram(2, &data)
        .expect("next request can refill");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    io::drain(0).expect("device completion");
    // A release event may be newly recorded; its producer has completed above.
    gpu.poll().expect("deferred pin release");
    io::drain(0).expect("release completion");
    if !gpu.unmap(1).expect("finished pin") {
        // Deterministic completion via the real driver, not a timeout or polling loop.
        let (_, context) = io::cuda::primary_context(0).expect("context");
        let _guard = io::cuda::CtxGuard::enter(context).expect("enter");
        io::cuda::check("cuCtxSynchronize", unsafe {
            (io::cuda::driver().expect("driver").ctx_synchronize)()
        })
        .expect("sync");
        assert!(gpu.unmap(1).expect("release now complete"));
    }
    gpu.fill_from_ram(1, &data)
        .expect("real TensorFS RAM refill");
    assert_eq!(bytes(&gpu.bind(1, 0).expect("refill bind")), data);
    let (_, context) = io::cuda::primary_context(0).expect("context");
    let _guard = io::cuda::CtxGuard::enter(context).expect("enter");
    io::cuda::check("cuCtxSynchronize", unsafe {
        (io::cuda::driver().expect("driver").ctx_synchronize)()
    })
    .expect("sync");
    gpu.poll().expect("release before export");
    io::cuda::check("cuCtxSynchronize", unsafe {
        (io::cuda::driver().expect("driver").ctx_synchronize)()
    })
    .expect("sync");
    let (spans, container) = gpu.export(1).expect("read-only page custody");
    drop(reserved);
    drop(gpu); // Ledger-held container retains the physical pages across a clean owner exit.
    let next = Gpu::open(0).expect("replacement owner");
    let reserved = next
        .reserve(data.len() as u64)
        .expect("replacement addresses");
    next.register(
        1,
        reserved,
        0,
        data.len() as u64,
        source,
        vec![item],
        0,
        "same-layout/encoding/transform/rank/device".into(),
    )
    .expect("replacement metadata");
    assert!(
        next.import(
            1,
            "same-layout/encoding/transform/rank/device",
            &spans,
            container
        )
        .expect("import")
    );
    let retained = next.bind(1, 0).expect("imported bind");
    assert_eq!(bytes(&retained), data);
    assert!(!next.unregister(1).expect("close defers a live view"));
    assert_eq!(bytes(&retained), data);
    drop(retained);
    next.wait_pending()
        .expect("wait for real consumer completion");
    next.poll().expect("retire the last view");
    assert!(next.unregister(1).expect("closed model now has no pages"));
    std::fs::remove_file(path).expect("remove source");
    println!("PASS independent pages, pin protection, RAM refill and kept-page import {uuid}");
}

fn one_device() -> String {
    let driver = io::cuda::driver().expect("real CUDA driver");
    let mut count = 0;
    io::cuda::check("cuDeviceGetCount", unsafe {
        (driver.device_get_count)(&mut count)
    })
    .expect("device inventory");
    assert_eq!(
        count, 1,
        "the public run must allocate exactly one visible GPU"
    );
    Gpu::open(0)
        .expect("real CUDA GPU")
        .uuid()
        .expect("GPU UUID")
}

fn test_page() -> (Arc<Gpu>, Vec<u8>) {
    let data: Vec<u8> = (0..2 * 1024 * 1024).map(|n| (n % 251) as u8).collect();
    let source = Arc::new(Source::files(vec![], false, 0));
    let gpu = Gpu::open(0).expect("real CUDA GPU");
    let reservation = gpu.reserve(data.len() as u64).expect("test address range");
    gpu.register(
        1,
        reservation,
        0,
        data.len() as u64,
        source,
        vec![Item {
            offset: 0,
            len: data.len() as u64,
            source: ItemSource::Inline(data.clone().into()),
            part_end: true,
        }],
        0,
        "slim-gpu-owned-view-test".into(),
    )
    .expect("register actual page");
    (gpu, data)
}

fn synchronize() {
    let (device, context) = io::cuda::primary_context(0).expect("test primary context");
    let guard = io::cuda::CtxGuard::enter(context).expect("test context entry");
    let driver = io::cuda::driver().expect("driver");
    let result = io::cuda::check("cuCtxSynchronize", unsafe { (driver.ctx_synchronize)() });
    drop(guard);
    io::cuda::check("cuDevicePrimaryCtxRelease", unsafe {
        (driver.primary_ctx_release)(device)
    })
    .expect("release test context reference");
    result.expect("consumer completion");
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn escaped_writable_view_retains_pages_and_recovers() {
    let uuid = one_device();
    let (gpu, data) = test_page();
    let held = std::sync::Mutex::new(None);
    let error = gpu
        .fill_with(1, |view| {
            *held.lock().expect("test slot") = Some(view);
            Ok(())
        })
        .expect_err("an escaped write view must refuse publication");
    assert!(
        !error.is_sticky(),
        "escaping a write view is a request failure"
    );
    assert!(!gpu.unmap(1).expect("live writable view blocks eviction"));
    gpu.recover()
        .expect("recovery keeps the live view protected");
    assert!(!gpu.unmap(1).expect("recovery cannot free a live view"));
    drop(held.lock().expect("test slot").take());
    gpu.poll().expect("deferred writable release");
    gpu.recover()
        .expect("same executor can recover after release");
    gpu.fill_from_ram(1, &data)
        .expect("healthy refill after escaped view release");
    let pin = gpu.bind(1, 0).expect("healthy read after recovery");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    synchronize();
    gpu.poll().expect("release read pin");
    synchronize();
    assert!(gpu.unregister(1).expect("no page remains after close"));
    assert!(gpu.facts().expect("closed state").is_empty());
    drop(gpu);

    // The escaped view may be the final owner after its Plane was closed. Its deleter must
    // finish the exact consumer streams, not free a page still in use on one of them.
    let (gpu, data) = test_page();
    let held = std::sync::Mutex::new(None);
    gpu.fill_with(1, |view| {
        *held.lock().expect("test slot") = Some(view);
        Ok(())
    })
    .expect_err("retained final view");
    let view = held
        .lock()
        .expect("test slot")
        .take()
        .expect("owned writable view");
    let weak = Arc::downgrade(&gpu);
    drop(gpu);
    assert!(
        weak.upgrade().is_some(),
        "the real allocation owner survives the Plane"
    );
    view.wait_on(0).expect("writable view stream");
    let mut read = vec![0; data.len()];
    unsafe {
        io::read(
            &[Copy {
                src: Src::Device {
                    device: 0,
                    ptr: view.ptr,
                    stream: 0,
                },
                dst: Dst::host(&mut read),
            }],
            &ReadOpts::default(),
        )
    }
    .expect("owned escaped bytes remain readable");
    assert_eq!(read, data);
    drop(view);
    assert!(
        weak.upgrade().is_none(),
        "last writable view releases its allocation owner"
    );
    println!("PASS escaped writable retention and same-process recovery {uuid}");
}

const READ_ONLY_WRITER: &std::ffi::CStr = c"\
.version 7.0
.target sm_70
.address_size 64
.visible .entry write_read_only(.param .u64 destination) {
    .reg .b64 %address;
    .reg .b32 %value;
    ld.param.u64 %address, [destination];
    mov.u32 %value, 7;
    st.global.u8 [%address], %value;
    ret;
}
";

#[test]
#[ignore = "helper invoked only by read_only_fault_is_contained_in_its_process"]
fn read_only_write_child() {
    assert_eq!(
        std::env::var("VARENA_GPU_HELPER").as_deref(),
        Ok("readonly")
    );
    let uuid = one_device();
    let (gpu, _) = test_page();
    gpu.fill(1).expect("fill and protect the page read-only");
    let pin = gpu.bind(1, 0).expect("bound read-only page");
    pin.wait_on(0).expect("ready page");
    synchronize();
    let (_, context) = io::cuda::primary_context(0).expect("child context");
    let _guard = io::cuda::CtxGuard::enter(context).expect("child context entry");
    let driver = io::cuda::driver().expect("driver");
    let mut module = std::ptr::null_mut();
    io::cuda::check("cuModuleLoadData", unsafe {
        (driver.module_load_data)(&mut module, READ_ONLY_WRITER.as_ptr().cast())
    })
    .expect("compile the one-store PTX kernel");
    let mut kernel = std::ptr::null_mut();
    io::cuda::check("cuModuleGetFunction", unsafe {
        (driver.module_get_function)(&mut kernel, module, c"write_read_only".as_ptr())
    })
    .expect("write kernel");
    let mut address = pin.ptr;
    let mut arguments = [(&mut address as *mut u64).cast::<std::ffi::c_void>()];
    let launched = unsafe {
        (driver.launch_kernel)(
            kernel,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            arguments.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    let code = if launched == io::cuda::SUCCESS {
        unsafe { (driver.ctx_synchronize)() }
    } else {
        launched
    };
    let error: varena::Error = io::cuda::check("write to read-only VMM page", code)
        .expect_err("GPU write protection must fault")
        .into();
    assert!(
        error.is_sticky(),
        "expected a real protection fault, got {error}"
    );
    println!("FAULT {uuid} {code}");
    std::io::stdout().flush().expect("fault result");
    // A sticky CUDA context belongs to this child. Process teardown releases it; running
    // destructors against the poisoned context would not test the healthy replacement.
    std::process::exit(0);
}

#[test]
#[ignore = "helper invoked only by read_only_fault_is_contained_in_its_process"]
fn healthy_replacement_child() {
    assert_eq!(std::env::var("VARENA_GPU_HELPER").as_deref(), Ok("healthy"));
    let uuid = one_device();
    let (gpu, data) = test_page();
    gpu.fill_from_ram(1, &data)
        .expect("fresh process refills the page from RAM");
    let pin = gpu.bind(1, 0).expect("replacement bind");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    synchronize();
    gpu.poll().expect("replacement release");
    synchronize();
    assert!(gpu.unregister(1).expect("replacement closes its page"));
    println!("HEALTHY {uuid}");
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn read_only_fault_is_contained_in_its_process() {
    let run = |test: &str, mode: &str, marker: &str| {
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--ignored",
                "--exact",
                test,
                "--nocapture",
                "--test-threads=1",
            ])
            .env("VARENA_GPU_HELPER", mode)
            .output()
            .expect("start isolated driver test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        print!("{stdout}{stderr}");
        assert!(output.status.success(), "{test} failed: {}", output.status);
        stdout
            .lines()
            .find_map(|line| {
                line.split_once(marker)
                    .map(|(_, tail)| tail.split_whitespace().next().expect("GPU UUID").to_owned())
            })
            .expect("actual child result marker")
    };
    let faulted = run("read_only_write_child", "readonly", "FAULT ");
    let healthy = run("healthy_replacement_child", "healthy", "HEALTHY ");
    assert_eq!(
        faulted, healthy,
        "replacement must run on the same physical GPU"
    );
    println!("PASS read-only VMM protection and healthy replacement {healthy}");
}

struct Pressure(Vec<u64>);
impl Drop for Pressure {
    fn drop(&mut self) {
        if let Ok(driver) = io::cuda::driver() {
            for pointer in self.0.drain(..) {
                unsafe {
                    (driver.mem_free)(pointer);
                }
            }
        }
    }
}

#[test]
#[ignore = "helper invoked only by cuda_oom_is_recovered_in_process"]
fn oom_recovery_child() {
    assert_eq!(std::env::var("VARENA_GPU_HELPER").as_deref(), Ok("oom"));
    let uuid = one_device();
    let (gpu, data) = test_page();
    let (device, context) = io::cuda::primary_context(0).expect("OOM context");
    let guard = io::cuda::CtxGuard::enter(context).expect("enter OOM context");
    let driver = io::cuda::driver().expect("driver");
    let mut pressure = Pressure(Vec::new());
    let mut free = 0;
    let mut total = 0;
    for _ in 0..64 {
        io::cuda::check("cuMemGetInfo", unsafe {
            (driver.mem_get_info)(&mut free, &mut total)
        })
        .expect("measured free VRAM");
        if free < data.len() {
            break;
        }
        let mut requested = free.saturating_sub(256 << 10);
        let mut allocated = false;
        for _ in 0..16 {
            if requested < 256 << 10 {
                break;
            }
            let mut pointer = 0;
            let code = unsafe { (driver.mem_alloc)(&mut pointer, requested) };
            if code == io::cuda::SUCCESS {
                pressure.0.push(pointer);
                allocated = true;
                break;
            }
            assert_eq!(
                code,
                io::cuda::ERROR_OUT_OF_MEMORY,
                "unexpected pressure allocation error"
            );
            requested /= 2;
        }
        if !allocated {
            break;
        }
    }
    io::cuda::check("cuMemGetInfo", unsafe {
        (driver.mem_get_info)(&mut free, &mut total)
    })
    .expect("remaining VRAM");
    let error = gpu
        .fill(1)
        .expect_err("real allocation pressure must exhaust the VMM page fill");
    assert!(error.is_oom(), "expected real CUDA OOM, got {error}");
    assert!(!error.is_sticky(), "CUDA OOM must remain recoverable");
    drop(pressure);
    drop(guard);
    io::cuda::check("cuDevicePrimaryCtxRelease", unsafe {
        (driver.primary_ctx_release)(device)
    })
    .expect("release OOM helper context");
    gpu.recover().expect("OOM recovery in the same process");
    gpu.fill_from_ram(1, &data)
        .expect("refill after releasing pressure");
    let pin = gpu.bind(1, 0).expect("recovered read");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    synchronize();
    gpu.poll().expect("release recovered pin");
    synchronize();
    assert!(gpu.unregister(1).expect("OOM recovery leaves no page"));
    assert!(gpu.facts().expect("closed state").is_empty());
    println!("PASS real CUDA OOM recovery {uuid} free_at_failure={free} total={total}");
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn cuda_oom_is_recovered_in_process() {
    let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--ignored",
            "--exact",
            "oom_recovery_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("VARENA_GPU_HELPER", "oom")
        .output()
        .expect("start OOM helper");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    print!("{stdout}{stderr}");
    assert!(output.status.success(), "OOM recovery child failed");
    assert!(
        stdout.contains("PASS real CUDA OOM recovery "),
        "missing actual OOM result"
    );
}

#[test]
#[ignore = "helper invoked only by killed_owner_keeps_read_only_pages"]
fn custody_holder_child() {
    use std::io::Read;
    use std::os::fd::FromRawFd;
    assert_eq!(std::env::var("VARENA_GPU_HELPER").as_deref(), Ok("holder"));
    let uuid = one_device();
    let mut control = unsafe { std::os::unix::net::UnixStream::from_raw_fd(0) };
    let (gpu, _) = test_page();
    gpu.fill(1).expect("holder fills read-only page");
    let (spans, container) = gpu.export(1).expect("holder exports queued page handles");
    println!("HOLDER {uuid}");
    std::io::stdout().flush().expect("holder identity");
    control
        .write_all(&(spans.len() as u32).to_le_bytes())
        .expect("write chunk count");
    for span in spans {
        control
            .write_all(&span.to_le_bytes())
            .expect("write chunk span");
    }
    platform::handles::send_fd(&control, &container).expect("send owned container");
    control
        .read_exact(&mut [0])
        .expect("holder waits until parent kills it");
    panic!("holder must be killed, never asked to exit cleanly");
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn killed_owner_keeps_read_only_pages() {
    use std::io::Read;
    use std::os::fd::OwnedFd;
    use std::os::unix::{net::UnixStream, process::ExitStatusExt};
    use std::process::Stdio;
    let (mut control, receiver) = UnixStream::pair().expect("custody control pair");
    let mut child = std::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--ignored",
            "--exact",
            "custody_holder_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("VARENA_GPU_HELPER", "holder")
        .stdin(Stdio::from(OwnedFd::from(receiver)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start holder");
    let mut count = [0; 4];
    control.read_exact(&mut count).expect("holder chunk count");
    let count = u32::from_le_bytes(count);
    assert!((1..=4096).contains(&count), "bounded nonempty export");
    let mut spans = Vec::new();
    for _ in 0..count {
        let mut span = [0; 8];
        control.read_exact(&mut span).expect("holder span");
        spans.push(u64::from_le_bytes(span));
    }
    let container = platform::handles::recv_fd(&control).expect("retained page container");
    child.kill().expect("SIGKILL holder");
    let output = child.wait_with_output().expect("reap killed holder");
    assert_eq!(
        output.status.signal(),
        Some(9),
        "holder must actually die from SIGKILL"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let holder = stdout
        .lines()
        .find_map(|line| line.split_once("HOLDER ").map(|(_, uuid)| uuid.to_owned()))
        .expect("holder physical GPU");
    let (gpu, data) = test_page();
    assert_eq!(gpu.uuid().expect("replacement GPU"), holder);
    io::metrics::take();
    assert!(
        gpu.import(1, "slim-gpu-owned-view-test", &spans, container)
            .expect("import after SIGKILL")
    );
    assert!(
        io::metrics::take().is_empty(),
        "import must not refill the source"
    );
    let pin = gpu.bind(1, 0).expect("imported page");
    assert_eq!(bytes(&pin), data);
    drop(pin);
    synchronize();
    gpu.poll().expect("release imported view");
    synchronize();
    assert!(gpu.unregister(1).expect("release killed-owner page"));
    println!("PASS SIGKILL read-only page custody {holder}");
}

struct ConsumerGate(std::sync::Arc<(std::sync::Mutex<(bool, bool)>, std::sync::Condvar)>);

impl ConsumerGate {
    fn install(stream: usize) -> Self {
        unsafe extern "C" fn hold(raw: *mut std::ffi::c_void) {
            let state = unsafe {
                Arc::from_raw(raw.cast::<(std::sync::Mutex<(bool, bool)>, std::sync::Condvar)>())
            };
            let mut flags = state
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            flags.0 = true;
            state.1.notify_all();
            while !flags.1 {
                flags = state
                    .1
                    .wait(flags)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
        let gate = Self(Arc::new((
            std::sync::Mutex::new((false, false)),
            std::sync::Condvar::new(),
        )));
        let raw = Arc::into_raw(gate.0.clone());
        let (device, context) = io::cuda::primary_context(0).expect("gate context");
        let guard = io::cuda::CtxGuard::enter(context).expect("gate context entry");
        let code = unsafe {
            (io::cuda::driver().expect("driver").launch_host_func)(
                stream as io::cuda::CUstream,
                hold,
                raw.cast_mut().cast(),
            )
        };
        if code != io::cuda::SUCCESS {
            drop(unsafe { Arc::from_raw(raw) });
        }
        drop(guard);
        io::cuda::check("cuDevicePrimaryCtxRelease", unsafe {
            (io::cuda::driver().expect("driver").primary_ctx_release)(device)
        })
        .expect("release gate context");
        io::cuda::check("cuLaunchHostFunc", code).expect("launch consumer gate");
        let flags = gate.0.0.lock().expect("gate state");
        let (flags, bound) = gate
            .0
            .1
            .wait_timeout_while(flags, std::time::Duration::from_secs(10), |flags| !flags.0)
            .expect("gate entry");
        assert!(
            !bound.timed_out() && flags.0,
            "consumer callback did not enter its test gate"
        );
        drop(flags);
        gate
    }
    fn release(&self) {
        let mut flags = self
            .0
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        flags.1 = true;
        self.0.1.notify_all();
    }
}
impl Drop for ConsumerGate {
    fn drop(&mut self) {
        self.release();
    }
}

#[test]
#[ignore = "requires an owned rental; never run on the local GPU"]
fn retired_page_releases_custody_after_last_view() {
    use std::os::fd::AsRawFd;
    let uuid = one_device();
    let (gpu, _) = test_page();
    gpu.fill(1).expect("fill retired page");
    let stream = io::cuda::Stream::new(0).expect("real consumer stream");
    let pin = gpu.bind(1, stream.raw()).expect("external read view");
    let (_, container) = gpu.export(1).expect("publish restart custody");
    let receiver = std::os::unix::net::UnixStream::from(container);
    let mut pending = 0_i32;
    assert_eq!(
        unsafe { libc::ioctl(receiver.as_raw_fd(), libc::FIONREAD, &mut pending) },
        0
    );
    assert!(pending > 0, "export must contain a real queued CUDA handle");
    let gate = ConsumerGate::install(stream.raw());
    assert!(!gpu.unregister(1).expect("retire with a live consumer"));
    let weak = Arc::downgrade(&gpu);
    drop(gpu);
    assert!(
        weak.upgrade().is_some(),
        "view retains the real allocation owner"
    );
    let releasing = std::thread::spawn(move || drop(pin));
    gate.release();
    releasing.join().expect("last-view teardown completes");
    assert!(
        weak.upgrade().is_none(),
        "the retired allocation owner must be gone"
    );
    assert_eq!(
        unsafe { libc::ioctl(receiver.as_raw_fd(), libc::FIONREAD, &mut pending) },
        0
    );
    assert_eq!(
        pending, 0,
        "retired page must revoke its queued custody handle"
    );
    receiver
        .set_nonblocking(true)
        .expect("nonblocking custody read");
    assert!(
        platform::handles::recv_fd(&receiver).is_err(),
        "no importable retired handle remains"
    );
    println!("PASS retired custody closes after last view completion {uuid}");
}
