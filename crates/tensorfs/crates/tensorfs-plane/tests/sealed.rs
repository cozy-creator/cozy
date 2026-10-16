//! Sealed host layouts on the real code path: the owner (the machine) fills one through the
//! verified read path and seals it; host-only planes (executors, no GPU) adopt it read-only.
//! Every check reads bytes back or asks the kernel.

mod common;

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use common::*;
use tensorfs_core::read;
use tensorfs_plane::host::{fill_sealed, reserve};
use tensorfs_plane::io::{IoTally, Source};
use tensorfs_plane::layout::Layout;
use tensorfs_plane::{Error, Tier};

fn sealed(fx: &Fixture, tally: &IoTally) -> OwnedFd {
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    fill_sealed("unet", &layout, &source, 3, tally, None).unwrap()
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

#[test]
fn a_reservation_made_before_the_plan_takes_any_layout_and_frees_what_it_does_not_need() {
    let fx = Fixture::new("reserve");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    for bytes in [layout.nbytes * 2, layout.nbytes / 3] {
        let reserved = reserve("unet", bytes, 3).unwrap();
        assert!(resident(reserved.as_raw_fd()) >= bytes.next_multiple_of(4096) * 9 / 10, "pages allocated up front");
        let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
        let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
        let fd = fill_sealed("unet", &layout, &source, 3, &IoTally::default(), Some(reserved)).unwrap();
        // Sized to the layout: a larger reservation's excess is freed, a smaller one grew.
        let size = std::fs::metadata(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap().len();
        assert!(size >= layout.nbytes && size < layout.nbytes + (1 << 20));
        assert!(resident(fd.as_raw_fd()) <= size);
        let plane = host_only();
        let ws = plane.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
        check_bytes(&plane, ws, None);
        plane.close().unwrap();
    }
}

fn open(fx: &Fixture) -> (OwnedFd, tensorfs_plane::host::OpenFill, Layout) {
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, fill) = tensorfs_plane::host::open_sealed("unet", &layout, None, None).unwrap();
    (fd, fill, layout)
}

#[test]
fn an_adopter_takes_a_layout_before_its_bytes_and_uses_each_region_once_ready() {
    let fx = Fixture::new("progressive");
    let (fd, fill, layout) = open(&fx);
    // Sealed from the start: no write, resize or new writable mapping, by anyone else.
    // SAFETY: scalar fcntl on a descriptor we hold.
    unsafe {
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) & (libc::F_SEAL_SHRINK | libc::F_SEAL_GROW), libc::F_SEAL_SHRINK | libc::F_SEAL_GROW);
    }
    let plane = host_only();
    let ws = plane.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    plane.set_pinned_budget(1 << 30).unwrap();
    let ticket = plane.want(ws, Tier::Pinned, None, 0, false).unwrap();
    assert!(!ticket.done().unwrap(), "nothing is Ready before the fill runs");
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let filler = std::thread::spawn(move || fill.run(&layout, &source, 2, &IoTally::default()));
    ticket.wait().unwrap();
    filler.join().unwrap().unwrap();
    check_bytes(&plane, ws, None);
    assert_eq!(plane.stats().host.counters.fill_bytes, 0, "waited for, never read here");
    plane.close().unwrap();
}

#[test]
fn a_failed_fill_reaches_every_adopter_as_an_error() {
    let fx = Fixture::new("progressive-fail");
    let (fd, fill, _) = open(&fx);
    let plane = host_only();
    let ws = plane.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    plane.set_pinned_budget(1 << 30).unwrap();
    let ticket = plane.want(ws, Tier::Pinned, None, 0, false).unwrap();
    drop(fill); // a filler that gives up poisons what it did not finish
    assert!(ticket.wait().is_err(), "a region its machine could not fill is an error, never read");
    // A later adopter refuses the failed layout outright.
    let other = host_only();
    assert!(other.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).is_err());
    other.close().unwrap();
    let _ = plane.close();
}

/// The object files a machine hands over: each object's verified store file, opened read-only.
fn object_files(fx: &Fixture) -> Vec<(tensorfs_core::ids::ObjectRef, std::fs::File)> {
    fx.objects.iter().map(|o| (o.clone(), std::fs::File::open(fx.store.blob_path(&o.sha256)).unwrap())).collect()
}

#[test]
fn a_layout_staged_in_part_holds_only_its_staged_regions_and_stages_more_on_request() {
    let fx = Fixture::new("staged");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    // Only region 0 is staged (the blocks that stream); 1 and 2 are holes.
    let (fd, fill) = tensorfs_plane::host::open_sealed("unet", &layout, None, Some(&[0])).unwrap();
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    fill.fill(&layout, &source, 2, &IoTally::default()).unwrap();
    let adopter = tensorfs_plane::host::HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    assert!(adopter.ready(0) && adopter.hole(1) && adopter.hole(2));
    let held = resident(fd.as_raw_fd());
    assert!(held < layout.regions[1].span, "the machine holds no copy of a hole: {held} B");

    // A pinned want covers the staged regions only; the holes are the files' to serve.
    let plane = host_only();
    let ws = plane.register_staged("unet", &fx.plan(), &regions(), fd.as_raw_fd(), object_files(&fx)).unwrap();
    plane.set_pinned_budget(1 << 30).unwrap();
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, Some(0));
    assert_eq!(plane.stats().host.used, layout.regions[0].span);

    // The executor's stream grew: the machine stages region 1 too, and only it.
    let taken = fill.take(&[1, 0]);
    assert_eq!(taken, [1], "region 0 was staged already");
    assert!(!adopter.hole(1) && !adopter.ready(1), "taken: adopters wait for it from now on");
    let staged = fill.stage(&taken, &layout, &source, 2, &IoTally::default()).unwrap();
    assert_eq!(staged, layout.regions[1].span);
    assert!(adopter.ready(1) && adopter.hole(2));
    plane.want(ws, Tier::Pinned, Some(&[1]), 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, Some(1));
    plane.close().unwrap();
    drop(fill); // holes stay holes when the filler goes; nothing staged is poisoned
    assert!(adopter.hole(2) && !adopter.failed(2) && !adopter.failed(1));
}

#[test]
fn a_files_source_reads_exactly_the_verified_bytes_it_was_handed() {
    let fx = Fixture::new("files-source");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let source = Source::files(object_files(&fx), true, 0);
    let tally = IoTally::default();
    for region in &layout.regions {
        let mut buf = vec![0u8; region.span as usize];
        for it in &region.items {
            // SAFETY: each item's range and padding lie inside `buf`; one thread.
            unsafe { source.read_item(it, buf.as_mut_ptr().add((it.offset - region.offset) as usize), tensorfs_plane::io::Mode::Cache, &tally) }.unwrap();
        }
        for p in layout.parts.iter().filter(|p| layout.regions[p.region as usize].offset == region.offset) {
            let key = p.what.trim_start_matches("unet/").trim_end_matches("#value");
            let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
            let at = (p.offset - region.offset) as usize;
            assert!(buf[at..at + n] == bytes_of(key, n)[..], "{} differs", p.what);
        }
    }
    // An object it was not handed is refused, never opened from anywhere else.
    let other = Source::files(Vec::new(), true, 0);
    let it = layout.regions[1].items.iter().find(|it| matches!(it.source, tensorfs_plane::layout::ItemSource::Object(_))).unwrap();
    let mut buf = vec![0u8; layout.regions[1].span as usize];
    assert!(unsafe { other.read_item(it, buf.as_mut_ptr(), tensorfs_plane::io::Mode::Cache, &tally) }.is_err());
}
