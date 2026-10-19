//! Streamed layouts (the disk rung) on the real code path: the machine's window stages regions
//! from a real store through the verified read path; adopters (other descriptions, as executors
//! are) read each region only through `with_region` and check every byte.
#![cfg(target_os = "linux")] // The host tier (memfd, seals, OFD locks) is Linux-only.

mod common;

use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use common::*;
use tensorfs::read;
use varena::host::HostMem;
use varena::io::{IoTally, Source};
use varena::layout::Layout;
use varena::window::open_window;
use varena::{Error, Tier};

/// Each region's expected bytes (padding excluded), by part.
fn expected(layout: &Layout) -> Vec<Vec<(usize, Vec<u8>)>> {
    (0..layout.regions.len())
        .map(|r| {
            layout
                .parts
                .iter()
                .filter(|p| p.region as usize == r)
                .map(|p| {
                    let key = p.what.trim_start_matches("unet/").trim_end_matches("#value");
                    let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
                    ((p.offset - layout.regions[r].offset) as usize, bytes_of(key, n))
                })
                .collect()
        })
        .collect()
}

fn read_region(host: &HostMem, r: u32, want: &[(usize, Vec<u8>)]) -> varena::Result<()> {
    read_holding(host, r, want, std::time::Duration::ZERO)
}

/// Check region `r`, hold it `hold`, and check it again: a refill under the claim shows.
fn read_holding(host: &HostMem, r: u32, want: &[(usize, Vec<u8>)], hold: std::time::Duration) -> varena::Result<()> {
    host.with_region(r, |p| {
        for pass in 0..1 + usize::from(!hold.is_zero()) {
            for (at, bytes) in want {
                // SAFETY: the region's bytes, held by our claim for the closure's duration.
                let got = unsafe { std::slice::from_raw_parts(p.add(*at), bytes.len()) };
                assert!(got == &bytes[..], "region {r} differs at {at} (pass {pass})");
            }
            thread::sleep(hold);
        }
        Ok(())
    })
}

struct Served {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<varena::Result<()>>>,
}
impl Served {
    fn end(mut self) -> varena::Result<()> {
        self.stop.store(true, Ordering::Release);
        self.thread.take().unwrap().join().unwrap()
    }
}

fn serve(fx: &Fixture, slots: usize) -> (std::os::fd::OwnedFd, Layout, Served) {
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, window) = open_window("unet", &layout, slots).unwrap();
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let stop = Arc::new(AtomicBool::new(false));
    let (l, s) = (layout.clone(), stop.clone());
    let thread = thread::spawn(move || window.serve(&l, &source, 3, &IoTally::default(), &s));
    (fd, layout, Served { stop, thread: Some(thread) })
}

fn resident(fd: i32) -> u64 {
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut st) }, 0);
    st.st_blocks as u64 * 512
}

#[test]
fn one_slot_streams_every_region_in_turn_and_again() {
    let fx = Fixture::new("window-one");
    fx.drop_cache();
    let (fd, layout, served) = serve(&fx, 1);
    let want = expected(&layout);
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    assert_eq!(host.window().unwrap().slots, 1);
    for r in [0u32, 1, 2, 0, 1, 2, 2, 1] {
        read_region(&host, r, &want[r as usize]).unwrap();
    }
    // Never more than the slot and the state page, though the layout is larger.
    let slot = host.window().unwrap().slot_span;
    assert!(resident(fd.as_raw_fd()) <= slot + (64 << 10), "{} > {slot}", resident(fd.as_raw_fd()));
    assert!(layout.nbytes > slot);
    drop(host);
    served.end().unwrap();
}

#[test]
fn concurrent_adopters_never_see_a_slot_refilled_under_them() {
    // Each refill waits a moment after the machine's snapshot of claims, so claims made in
    // between are the norm: only the check after clearing a slot's word keeps them safe.
    #[cfg(feature = "faults")]
    varena::faults::arm_stall("window-victim", 3);
    let fx = Fixture::new("window-race");
    let (fd, layout, served) = serve(&fx, 2);
    let want = Arc::new(expected(&layout));
    let mut readers = Vec::new();
    for a in 0..2u32 {
        // Two descriptions (two executors), each with three reading threads that hold what they
        // read a while and check it again: wants overlap, so the machine refills slots while
        // others read.
        let host = Arc::new(HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap());
        for t in 0..3u32 {
            let (host, want) = (host.clone(), want.clone());
            readers.push(thread::spawn(move || {
                let mut x = 17 + 31 * a + 7 * t;
                for _ in 0..20 {
                    x = x.wrapping_mul(1103515245).wrapping_add(12345);
                    let r = (x >> 16) % 3;
                    read_holding(&host, r, &want[r as usize], std::time::Duration::from_millis(2)).unwrap();
                }
            }));
        }
    }
    for r in readers {
        r.join().unwrap();
    }
    served.end().unwrap();
}

#[test]
fn a_region_the_machine_cannot_read_fails_its_readers_and_only_them() {
    let fx = Fixture::new("window-gone");
    let (fd, layout, served) = serve(&fx, 2);
    // The object behind region 2 (c.weight) vanishes after the lease (a disk fault): its stage
    // fails, the others still stream.
    let object = layout.regions[2].items.iter().find_map(|it| match &it.source {
        varena::layout::ItemSource::Object(o) => Some(o.obj.sha256.clone()),
        _ => None,
    });
    std::fs::remove_file(fx.store.blob_path(&object.unwrap())).unwrap();
    let want = expected(&layout);
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    read_region(&host, 0, &want[0]).unwrap();
    assert!(read_region(&host, 2, &want[2]).is_err(), "a region its machine could not read is never read");
    read_region(&host, 1, &want[1]).unwrap();
    served.end().unwrap();
}

#[test]
fn a_window_that_stops_serving_fails_its_waiting_readers() {
    let fx = Fixture::new("window-stop");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, window) = open_window("unet", &layout, 1).unwrap();
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    let waiter = thread::spawn(move || host.with_region(1, |_| Ok(())).map(|_| ()));
    thread::sleep(std::time::Duration::from_millis(50));
    drop(window); // its machine gave up before staging anything
    assert!(waiter.join().unwrap().is_err());
}

#[test]
fn a_plane_registers_a_streamed_layout_but_never_pins_it() {
    let fx = Fixture::new("window-plane");
    let (fd, _, served) = serve(&fx, 1);
    let plane = host_only();
    let ws = plane.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    plane.set_pinned_budget(1 << 30).unwrap();
    assert!(matches!(plane.want(ws, Tier::Pinned, None, 0, false), Err(Error::BelowFloor { .. })));
    assert!(matches!(plane.want(ws, Tier::Pinned, None, 0, true), Err(Error::BelowFloor { .. })));
    assert_eq!(plane.stats().host.used, 0);
    // A window of another layout is refused at registration.
    let mut other = regions();
    other.swap(0, 2);
    assert!(plane.register_sealed("unet", &fx.plan(), &other, fd.as_raw_fd()).is_err());
    plane.close().unwrap();
    served.end().unwrap();
}

/// A layout of one region in a window of one slot has exactly the layout's size: the state
/// page's magic, not the size, tells them apart.
#[test]
fn a_one_region_window_the_size_of_its_layout_is_still_a_window() {
    let fx = Fixture::new("window-single");
    let one = vec![regions().concat()];
    let layout = Layout::build(&fx.plan(), &one).unwrap();
    let (fd, window) = open_window("unet", &layout, 1).unwrap();
    let (full, _filling) = varena::host::open_sealed("unet", &layout, None, None).unwrap();
    let size = |fd: i32| std::fs::File::from(fd_dup(fd)).metadata().unwrap().len();
    assert_eq!(size(fd.as_raw_fd()), size(full.as_raw_fd()), "the case this test is about");
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &layout).unwrap();
    assert!(host.window().is_some());
    assert!(HostMem::adopt_sealed(full.as_raw_fd(), &layout).unwrap().window().is_none());
    drop(window);
}

fn fd_dup(fd: i32) -> std::os::fd::OwnedFd {
    // SAFETY: dup of a live descriptor we hold; the copy is ours.
    unsafe { <std::os::fd::OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(libc::dup(fd)) }
}
