//! Sealed host layouts on the real code path: the owner (the machine) fills one through the
//! verified read path and seals it; executors adopt it read-only.
//! Every check reads bytes back or asks the kernel.
#![cfg(target_os = "linux")] // The host tier (memfd, seals, OFD locks) is Linux-only.

mod common;

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use common::*;
use tensorfs_core::read;
use varena::host::{fill_sealed, reserve, HostMem};
use varena::io::{IoTally, Source};
use varena::layout::Layout;
use varena::Error;

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
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    check_host(&HostMem::adopt_sealed(raw, &layout).unwrap(), &layout, None);

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
}

#[test]
fn only_a_sealed_complete_layout_is_adopted_read_only() {
    let fx = Fixture::new("unsealed");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    // A memfd anyone may still write is refused.
    let open = HostMem::create("unet", &layout).unwrap();
    assert_eq!(varena::host::sealed_regions(open.fd()).unwrap(), regions(), "every layout's state page names its regions");
    let empty = varena::host::reserve("empty", 4096, 1).unwrap();
    assert!(matches!(varena::host::sealed_regions(empty.as_raw_fd()), Err(Error::Invalid(_))), "a memfd without a state page names none");
    match HostMem::adopt_sealed(open.fd(), &layout) {
        Err(Error::Invalid(_)) => {}
        r => panic!("an unsealed memfd must be refused, got {:?}", r.err()),
    }
    // A sealed layout of another grouping is refused, not reinterpreted.
    let fd = sealed(&fx, &IoTally::default());
    let mut moved = regions();
    let tail = moved[2].pop().unwrap();
    moved[0].push(tail);
    assert!(HostMem::adopt_sealed(fd.as_raw_fd(), &Layout::build(&fx.plan(), &moved).unwrap()).is_err());
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
        check_host(&HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap(), &layout, None);
    }
}

fn open(fx: &Fixture) -> (OwnedFd, varena::host::OpenFill, Layout) {
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, fill) = varena::host::open_sealed("unet", &layout, None, None).unwrap();
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
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    assert!(!host.ready(0), "nothing is Ready before the fill runs");
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let filling = layout.clone();
    let filler = std::thread::spawn(move || fill.run(&filling, &source, 2, &IoTally::default()));
    for r in 0..layout.regions.len() as u32 {
        host.wait_ready(r).unwrap();
    }
    filler.join().unwrap().unwrap();
    check_host(&host, &layout, None);
}

#[test]
fn a_failed_fill_reaches_every_adopter_as_an_error() {
    let fx = Fixture::new("progressive-fail");
    let (fd, fill, layout) = open(&fx);
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    drop(fill); // a filler that gives up poisons what it did not finish
    assert!(host.wait_ready(0).is_err(), "a region its machine could not fill is an error, never read");
    // A later adopter refuses the failed layout outright.
    assert!(HostMem::adopt_sealed(fd.as_raw_fd(), &layout).is_err());
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
    let (fd, fill) = varena::host::open_sealed("unet", &layout, None, Some(&[0])).unwrap();
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    fill.fill(&layout, &source, 2, &IoTally::default()).unwrap();
    let adopter = varena::host::HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    assert!(adopter.ready(0) && adopter.hole(1) && adopter.hole(2));
    let held = resident(fd.as_raw_fd());
    assert!(held < layout.regions[1].span, "the machine holds no copy of a hole: {held} B");

    check_host(&adopter, &layout, Some(0));

    // The executor's stream grew: the machine stages region 1 too, and only it.
    let taken = fill.take(&[1, 0]);
    assert_eq!(taken, [1], "region 0 was staged already");
    assert!(!adopter.hole(1) && !adopter.ready(1), "taken: adopters wait for it from now on");
    let staged = fill.stage(&taken, &layout, &source, 2, &IoTally::default()).unwrap();
    assert_eq!(staged, layout.regions[1].span);
    assert!(adopter.ready(1) && adopter.hole(2));
    check_host(&adopter, &layout, Some(1));
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
            unsafe { source.read_item(it, buf.as_mut_ptr().add((it.offset - region.offset) as usize), varena::io::Mode::Cache, &tally) }.unwrap();
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
    let it = layout.regions[1].items.iter().find(|it| matches!(it.source, varena::layout::ItemSource::Object(_))).unwrap();
    let mut buf = vec![0u8; layout.regions[1].span as usize];
    assert!(unsafe { other.read_item(it, buf.as_mut_ptr(), varena::io::Mode::Cache, &tally) }.is_err());
}
