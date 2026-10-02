//! Sealed host layouts on the real code path: the owner (the machine) fills one through the
//! verified read path and seals it; host-only planes (executors, no GPU) adopt it read-only.
//! Every check reads bytes back or asks the kernel.

mod common;

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use common::*;
use tensorfs_core::read;
use tensorfs_plane::host::fill_sealed;
use tensorfs_plane::io::{IoTally, Source};
use tensorfs_plane::layout::Layout;
use tensorfs_plane::{Error, Tier};

fn sealed(fx: &Fixture, tally: &IoTally) -> OwnedFd {
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    fill_sealed("unet", &layout, &source, 3, tally).unwrap()
}

fn resident(fd: i32) -> u64 {
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut st) }, 0);
    st.st_blocks as u64 * 512
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap()
}

#[test]
fn a_sealed_layout_holds_every_byte_and_nobody_can_change_one() {
    let fx = Fixture::new("sealed");
    fx.drop_cache();
    let tally = IoTally::default();
    let fd = sealed(&fx, &tally);
    let raw = fd.as_raw_fd();
    let payload: u64 = TENSORS.iter().map(|(_, n)| *n as u64).sum();
    let read = |n: &std::sync::atomic::AtomicU64| n.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(read(&tally.cached_bytes) + read(&tally.direct_bytes) + read(&tally.buffered_bytes) + read(&tally.inline_bytes), payload);
    assert!(read(&tally.direct_bytes) > payload / 2, "a cold fill reads by O_DIRECT");
    assert!(resident(raw) >= payload);

    // The bytes, through an adopter's own read-only view.
    let plane = host_only();
    let ws = plane.register_sealed("unet", &fx.plan(), &regions(), raw).unwrap();
    check_bytes(&plane, ws, None);

    // Writes, resizes, punches and new writable mappings all refused, also through a new
    // read-write open of the same memfd.
    // SAFETY: plain syscalls on descriptors we own; each is expected to fail.
    unsafe {
        let rw = libc::open(format!("/proc/self/fd/{raw}\0").as_ptr().cast(), libc::O_RDWR | libc::O_CLOEXEC);
        assert!(rw >= 0);
        let rw = OwnedFd::from_raw_fd(rw);
        assert_eq!(libc::pwrite(rw.as_raw_fd(), b"x".as_ptr().cast(), 1, 0), -1);
        assert_eq!(errno(), libc::EPERM);
        assert_ne!(libc::ftruncate(rw.as_raw_fd(), 0), 0);
        assert_ne!(libc::fallocate(rw.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, 0, 4096), 0);
        let map = libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, rw.as_raw_fd(), 0);
        assert_eq!(map, libc::MAP_FAILED);
        assert_eq!(libc::fcntl(raw, libc::F_ADD_SEALS, libc::F_SEAL_GROW), -1); // sealing is sealed too
        assert_eq!(errno(), libc::EPERM);
    }
    plane.close().unwrap();
}

#[test]
fn an_adopter_pins_inside_its_budget_reads_nothing_and_owns_no_ram() {
    let fx = Fixture::new("adopt-sealed");
    let fd = sealed(&fx, &IoTally::default());
    let full = resident(fd.as_raw_fd());
    let plane = host_only();
    let ws = plane.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let spans: Vec<u64> = plane.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    // Nothing joins the books at registration: only what a budget wants.
    assert_eq!(plane.stats().host.used, 0);
    plane.set_pinned_budget(spans[0] + spans[1]).unwrap();
    plane.want(ws, Tier::Pinned, Some(&[0, 1]), 0, false).unwrap().wait().unwrap();
    let s = plane.stats();
    assert_eq!(s.host.used, spans[0] + spans[1]);
    assert_eq!(s.host.counters.fill_bytes, 0, "adoption reads no store");
    assert_eq!(s.host.direct_bytes + s.host.cached_bytes + s.host.buffered_bytes, 0);
    assert_eq!(s.host.memfd_bytes, 0, "the sealed layout's RAM is its owner's");
    assert_eq!(s.host.sealed_bytes, full);
    // A budget cut lets regions go from the books but never punches: the bytes stay.
    let t = plane.set_pinned_budget(0).unwrap();
    assert_eq!(t.freed, spans[0] + spans[1]);
    assert_eq!(resident(fd.as_raw_fd()), full);
    check_bytes(&plane, ws, None);
    // Every region is Ready for good: a want past the budget is refused, never filled.
    match plane.want(ws, Tier::Pinned, Some(&[2]), 0, false) {
        Err(Error::BelowFloor { .. }) => {}
        other => panic!("expected BelowFloor, got {:?}", other.err()),
    }
    // Closing the adopter leaves the layout whole for the next one.
    plane.close().unwrap();
    assert_eq!(resident(fd.as_raw_fd()), full);
    let next = host_only();
    let again = next.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    check_bytes(&next, again, None);
    next.close().unwrap();
}

#[test]
fn only_a_sealed_complete_layout_is_adopted_read_only() {
    let fx = Fixture::new("unsealed");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    // A pinned tier an executor filled is Ready but not sealed: refused.
    let other = host_only();
    match other.register_sealed("unet", &fx.plan(), &regions(), plane.host_fd(ws).unwrap()) {
        Err(Error::Invalid(_)) => {}
        r => panic!("an unsealed memfd must be refused, got {:?}", r.err()),
    }
    // A sealed layout of another grouping is refused, not reinterpreted.
    let fd = sealed(&fx, &IoTally::default());
    let mut moved = regions();
    let tail = moved[2].pop().unwrap();
    moved[0].push(tail);
    assert!(other.register_sealed("unet", &fx.plan(), &moved, fd.as_raw_fd()).is_err());
    other.close().unwrap();
    plane.close().unwrap();
}
