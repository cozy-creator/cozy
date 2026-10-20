//! Small mapped-host mechanism proof; host admission itself is owned by varena's service.
#![cfg(unix)]

use std::ffi::{c_void, CString};
use std::sync::{mpsc, Arc};
use std::sync::atomic::{AtomicUsize, Ordering};
use tensorfs::io::{self, cuda, Copy, Dst, MappedBuffer, ReadOpts, Src};

#[test]
fn host_charge_rounds_to_actual_requested_pages_without_initializing_cuda() {
    let page = platform::shm::page_size();
    assert_eq!(MappedBuffer::allocation_size(0).unwrap(), 0);
    assert_eq!(MappedBuffer::allocation_size(1).unwrap(), page);
    assert_eq!(MappedBuffer::allocation_size(page).unwrap(), page);
    assert_eq!(MappedBuffer::allocation_size(page + 1).unwrap(), 2 * page);
    assert!(MappedBuffer::allocation_size(usize::MAX).is_err());
}

struct Reservation(Arc<AtomicUsize>);
impl Drop for Reservation {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}

struct Gate { started: mpsc::Sender<()>, release: mpsc::Receiver<()> }
unsafe extern "C" fn gate(raw: *mut c_void) {
    let gate = unsafe { Box::from_raw(raw.cast::<Gate>()) };
    let _ = gate.started.send(());
    let _ = gate.release.recv();
}

#[test]
#[ignore = "requires an allocated CUDA GPU; no mock driver"]
fn mapped_host_uva_and_final_alias_keep_the_reservation_through_side_stream_use() {
    let dropped = Arc::new(AtomicUsize::new(0));
    let stream = cuda::Stream::new(0).unwrap();
    let buffer = Arc::new(MappedBuffer::new(
        0, 4096, stream.raw(), Arc::new(Reservation(dropped.clone())),
    ).unwrap().expect("assigned CUDA device must support mapped host memory"));
    assert_eq!(buffer.allocated(), MappedBuffer::allocation_size(buffer.len()).unwrap());
    unsafe { buffer.host_ptr().cast::<u32>().write(7) };
    let output = io::DeviceBuffer::new(0, 4).unwrap();
    let driver = cuda::driver().unwrap();
    let _ctx = cuda::CtxGuard::enter(buffer.context()).unwrap();
    let ptx = CString::new(r#"
.version 7.0
.target sm_50
.address_size 64
.visible .entry mapped_read(.param .u64 src, .param .u64 dst) {
    .reg .b64 %source, %destination;
    .reg .b32 %value;
    ld.param.u64 %source, [src];
    ld.param.u64 %destination, [dst];
    ld.global.u32 %value, [%source];
    add.u32 %value, %value, 1;
    st.global.u32 [%destination], %value;
    ret;
}
"#).unwrap();
    let mut module = std::ptr::null_mut();
    cuda::check("cuModuleLoadData", unsafe { (driver.module_load_data)(&mut module, ptx.as_ptr().cast()) }).unwrap();
    let mut function = std::ptr::null_mut();
    cuda::check("cuModuleGetFunction", unsafe {
        (driver.module_get_function)(&mut function, module, c"mapped_read".as_ptr())
    }).unwrap();
    let (started, entered) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let hold = Box::into_raw(Box::new(Gate { started, release: blocked }));
    cuda::check("cuLaunchHostFunc", unsafe {
        (driver.launch_host_func)(stream.raw() as _, gate, hold.cast())
    }).unwrap();
    let mut src = buffer.ptr();
    let mut dst = output.ptr();
    let mut args = [(&mut src as *mut u64).cast(), (&mut dst as *mut u64).cast()];
    cuda::check("cuLaunchKernel", unsafe {
        (driver.launch_kernel)(function, 1, 1, 1, 1, 1, 1, 0, stream.raw() as _, args.as_mut_ptr(), std::ptr::null_mut())
    }).unwrap();
    entered.recv().unwrap();
    let alias = Arc::clone(&buffer);
    let weak = Arc::downgrade(&buffer);
    drop(buffer);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    let retiring = std::thread::spawn(move || drop(alias));
    while weak.strong_count() != 0 { std::thread::yield_now(); }
    // The stream's callback still holds the kernel behind the gate. No last-alias
    // release may give its actual host lease back before that consumer finishes.
    let retained = dropped.load(Ordering::SeqCst) == 0;
    release.send(()).unwrap();
    retiring.join().unwrap();
    assert!(retained, "host reservation escaped before the side-stream consumer");
    let mut bytes = [0u8; 4];
    unsafe { io::read(&[Copy {
        src: Src::Device { device: 0, ptr: output.ptr(), stream: stream.raw() },
        dst: Dst::host(&mut bytes),
    }], &ReadOpts::default()) }.unwrap();
    assert_eq!(u32::from_ne_bytes(bytes), 8);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    cuda::check("cuModuleUnload", unsafe { (driver.module_unload)(module) }).unwrap();
    println!("PASS mapped host UVA and final alias preserve the reservation {}", cuda::uuid(stream.device()).unwrap());
}
