//! Device paths on a real GPU: fills from chunk files (O_DIRECT cold, page cache warm) and from a
//! sealed layout, drops that give memory back to the driver, `DeviceFull`/`Floor`, leases as the
//! only pin, the streaming ring under a slow consumer, custody, and the copy path's faults.
//! Ignored by default; one GPU operator at a time:
//! `cargo test -p varena --features faults --test gpu -- --ignored --test-threads 1`
#![cfg(target_os = "linux")]

mod common;

use std::ffi::c_void;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd};

use common::*;
use varena::cuda::{self, CUstream, CtxGuard};
use varena::{Error, Host, Lease, Plane, PlaneConfig, WsId};

const DEV: i32 = 0;
const PAGE: Host = Host::PageCache { direct: true };

fn plane() -> Plane {
    open(PlaneConfig { devices: vec![DEV], readers: 4, staging_buffers: 3, staging_bytes: 16 << 20, ..Default::default() })
}

fn open(cfg: PlaneConfig) -> Plane {
    Plane::open(cfg).unwrap()
}

/// A consumer stream in the device's primary context (what the executor's framework uses).
struct Stream(CUstream, #[allow(dead_code)] CtxGuard);
impl Stream {
    fn new() -> Stream {
        let (_, ctx) = cuda::primary_context(DEV).unwrap();
        let g = CtxGuard::enter(ctx).unwrap();
        let d = cuda::driver().unwrap();
        let mut s: CUstream = std::ptr::null_mut();
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) }).unwrap();
        Stream(s, g)
    }
    fn raw(&self) -> u64 {
        self.0 as u64
    }
    /// Device bytes at `ptr`, read on this stream (so after any waits enqueued on it).
    fn read(&self, ptr: u64, n: u64) -> Vec<u8> {
        let d = cuda::driver().unwrap();
        let mut v = vec![0u8; n as usize];
        cuda::check("cuMemcpyDtoHAsync", unsafe { (d.memcpy_dtoh_async)(v.as_mut_ptr() as *mut c_void, ptr, n as usize, self.0) }).unwrap();
        self.sync();
        v
    }
    /// Queue a copy of device bytes into pinned host memory on this stream; no host wait.
    fn read_async(&self, ptr: u64, n: u64) -> Pinned {
        let d = cuda::driver().unwrap();
        let mut h: *mut c_void = std::ptr::null_mut();
        cuda::check("cuMemHostAlloc", unsafe { (d.mem_host_alloc)(&mut h, n as usize, 0) }).unwrap();
        cuda::check("cuMemcpyDtoHAsync", unsafe { (d.memcpy_dtoh_async)(h, ptr, n as usize, self.0) }).unwrap();
        Pinned(h as *mut u8, n as usize)
    }
    fn sync(&self) {
        let d = cuda::driver().unwrap();
        cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(self.0) }).unwrap();
    }
    /// Stall this stream for `ms` on the device timeline: a stand-in for a long kernel.
    fn stall(&self, ms: u64) {
        extern "C" fn sleep(p: *mut c_void) {
            std::thread::sleep(std::time::Duration::from_millis(p as u64));
        }
        let d = cuda::driver().unwrap();
        cuda::check("cuLaunchHostFunc", unsafe { (d.launch_host_func)(self.0, sleep, ms as *mut c_void) }).unwrap();
    }
}

struct Pinned(*mut u8, usize);
impl Pinned {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, self.1) }
    }
}
impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { (cuda::driver().unwrap().mem_free_host)(self.0 as *mut c_void) };
    }
}

/// (free, total) device memory from the driver.
fn mem_info() -> (u64, u64) {
    let (_, ctx) = cuda::primary_context(DEV).unwrap();
    let _g = CtxGuard::enter(ctx).unwrap();
    varena::device::mem_info().unwrap()
}

/// A lease on region `r` in place, its bytes checked.
fn resident(p: &Plane, ws: WsId, s: &Stream, r: u32) -> Lease {
    let l = p.acquire(ws, DEV, r, s.raw()).unwrap();
    check_region(&p.layout(ws).unwrap(), r, &s.read(l.ptr, l.nbytes));
    l
}

fn all_exact(p: &Plane, ws: WsId, s: &Stream) {
    for r in 0..p.layout(ws).unwrap().regions.len() as u32 {
        resident(p, ws, s, r).release(s.raw()).unwrap();
    }
}

#[test]
#[ignore = "GPU"]
fn fills_from_chunk_files_are_exact_cold_and_warm() {
    let fx = Fixture::new("gpu-files");
    let s = Stream::new();
    for (direct, cold) in [(false, true), (true, false), (true, true)] {
        let p = plane();
        let ws = p.register(&fx.ck, &fx.regions(), Host::PageCache { direct }).unwrap();
        if cold {
            fx.drop_cache();
        }
        p.want(ws, DEV, None).unwrap().wait().unwrap();
        all_exact(&p, ws, &s);
        let st = p.stats();
        let d = &st.devices[0].counters;
        let total: u64 = p.layout(ws).unwrap().regions.iter().map(|r| r.nbytes).sum();
        println!("direct {direct} cold {cold}: {:?}; disk copy {} B at {:.1} GB/s", st.io, d.disk_copy_bytes, d.disk_copy_bytes as f64 / d.disk_copy_ns.max(1) as f64);
        assert_eq!(d.misses, 0, "a want leaves nothing to miss");
        assert!(d.disk_copy_bytes >= total);
        match (direct, cold) {
            (true, true) => assert!(st.io.direct_bytes > 0 || st.io.direct_refused > 0, "{:?}", st.io),
            (true, false) => assert_eq!(st.io.direct_bytes, 0, "a resident chunk is copied from the page cache"),
            _ => assert_eq!(st.io.direct_bytes, 0),
        }
        assert!(st.regions.iter().all(|r| r.state == "ready" && r.last_use > 0 && r.live == 0));
        p.close().unwrap();
    }
}

#[test]
#[ignore = "GPU"]
fn a_sealed_layout_copies_by_one_dma_and_its_holes_from_the_files() {
    let fx = Fixture::new("gpu-sealed");
    let layout = varena::Layout::build(fx.ck.header(), &fx.regions()).unwrap();
    let (fd, fill) = sealed(&fx, &layout, Some(&[0, 1]));
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), Host::Sealed(fd.as_raw_fd())).unwrap();
    let s = Stream::new();
    p.want(ws, DEV, None).unwrap().wait().unwrap();
    all_exact(&p, ws, &s);
    let st = p.stats();
    let d = &st.devices[0].counters;
    let held = layout.regions[0].nbytes + layout.regions[1].nbytes;
    let holes = layout.regions[2].nbytes + layout.regions[3].nbytes;
    println!("sealed: locked {} B at {:.1} GB/s, bounced {} B, files {} B; register_failed {}", d.host_copy_bytes, d.host_copy_bytes as f64 / d.host_copy_ns.max(1) as f64, d.bounce_copy_bytes, d.disk_copy_bytes, st.io.register_failed);
    assert_eq!(d.host_copy_bytes + d.bounce_copy_bytes, held);
    assert_eq!(d.disk_copy_bytes, holes);
    if st.io.register_failed == 0 {
        assert_eq!(d.bounce_copy_bytes, 0, "never pageable: locked, or bounced only when refused");
    }
    // The owner stages a hole later: from then on it copies from the layout.
    p.drop_regions(ws, DEV, None).unwrap();
    assert_eq!(fill.take(&[2]), vec![2]);
    let region = &layout.regions[2];
    for c in &region.chunks {
        let bytes = std::fs::read(&fx.ck.chunk_paths()[c.index as usize]).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), fill.base().add(c.offset as usize), bytes.len()) };
    }
    fill.set_ready(2);
    resident(&p, ws, &s, 2).release(s.raw()).unwrap();
    let d2 = p.stats().devices[0].counters.clone();
    assert_eq!(d2.disk_copy_bytes, holes, "a staged region no longer reads the files");
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn drop_gives_the_memory_back_to_the_driver_and_only_a_lease_keeps_a_region() {
    let fx = Fixture::new("gpu-drop");
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    let (free0, _) = mem_info();
    p.want(ws, DEV, None).unwrap().wait().unwrap();
    let (free1, _) = mem_info();
    assert!(free0 - free1 >= spans.iter().sum::<u64>(), "mapping takes driver memory");
    assert_eq!(p.stats().devices[0].mapped, spans.iter().sum::<u64>());
    // A leased region stays; the rest go back to the driver at once (no idle pool).
    let held = p.acquire(ws, DEV, 1, s.raw()).unwrap();
    let freed = p.drop_regions(ws, DEV, None).unwrap();
    assert_eq!(freed, spans.iter().sum::<u64>() - spans[1]);
    let (free2, _) = mem_info();
    assert!(free2 >= free1 + freed, "free memory rises by the dropped bytes: {free1} -> {free2}, dropped {freed}");
    let st = p.stats();
    assert_eq!((st.devices[0].mapped, st.devices[0].leased), (spans[1], spans[1]));
    assert_eq!(st.regions.iter().map(|r| (r.region, r.live)).collect::<Vec<_>>(), vec![(1, 1)]);
    held.release(s.raw()).unwrap();
    assert_eq!(p.drop_regions(ws, DEV, Some(&[1])).unwrap(), spans[1]);
    assert_eq!(p.stats().devices[0].mapped, 0);
    // A drop of a region still filling waits for its copy, then unmaps it.
    fx.drop_cache();
    let t = p.want(ws, DEV, Some(&[1])).unwrap();
    assert_eq!(p.drop_regions(ws, DEV, Some(&[1])).unwrap(), spans[1]);
    assert!(t.wait().is_err(), "the ticket reports the dropped region");
    // An unmapped region is a miss: the next acquire fills it again.
    resident(&p, ws, &s, 1).release(s.raw()).unwrap();
    assert_eq!(p.stats().devices[0].counters.misses, 1);
    s.sync();
    assert!(p.stats().poisoned.is_none());
    p.close().unwrap();
    let (free3, _) = mem_info();
    assert!(free3 + (16 << 20) >= free0, "closing gives everything back: {free0} -> {free3}");
}

#[test]
#[ignore = "GPU"]
fn device_full_and_floor_refuse_and_map_nothing() {
    let fx = Fixture::new("gpu-full");
    let (free, total) = mem_info();
    // A floor above the device's memory: nothing could ever fit.
    let p = open(PlaneConfig { devices: vec![DEV], floor_bytes: total, ..Default::default() });
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    match p.want(ws, DEV, None) {
        Err(Error::Floor { need, capacity: 0 }) => assert_eq!(need, p.layout(ws).unwrap().nbytes),
        other => panic!("expected Floor, got {:?}", other.err()),
    }
    p.close().unwrap();
    // A floor that leaves 32 MiB of today's free memory: the 96 MiB want does not fit now.
    let room = 32 << 20;
    let p = open(PlaneConfig { devices: vec![DEV], floor_bytes: free - room, ..Default::default() });
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let need = p.layout(ws).unwrap().nbytes;
    match p.want(ws, DEV, None) {
        Err(Error::DeviceFull { deficit }) => assert!(deficit >= need - room && deficit < need + (64 << 20), "{deficit} for {need}"),
        other => panic!("expected DeviceFull, got {:?}", other.err()),
    }
    let st = p.stats();
    assert_eq!((st.devices[0].mapped, st.regions.len()), (0, 0), "all or nothing");
    // A region that fits is still served.
    let s = Stream::new();
    resident(&p, ws, &s, 3).release(s.raw()).unwrap();
    assert!(matches!(p.stream(ws, DEV, &[1], 1, None), Err(Error::DeviceFull { .. })), "a ring maps under the same rule");
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn a_cursor_serves_mapped_regions_in_place_but_never_pins_them() {
    let fx = Fixture::new("gpu-home");
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    p.want(ws, DEV, Some(&[0])).unwrap().wait().unwrap();
    let cur = p.stream(ws, DEV, &[0, 1, 2, 3], 2, None).unwrap();
    assert_eq!(cur.home, vec![true, false, false, false]);
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    assert_eq!(p.drop_regions(ws, DEV, Some(&[0])).unwrap(), spans[0], "only a lease pins");
    let mut reads = Vec::new();
    for _ in 0..2 {
        for r in 0..4u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            assert_eq!(l.ring_offset.is_none(), r == 0);
            s.stall(10);
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
    }
    s.sync();
    let layout = p.layout(ws).unwrap();
    for (r, buf) in &reads {
        check_region(&layout, *r, buf.bytes());
    }
    assert_eq!(p.stats().devices[0].counters.misses, 1, "the dropped home region refilled once");
    cur.close().unwrap();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn the_ring_reuses_bytes_only_after_their_reader_released_them() {
    let fx = Fixture::new("gpu-ring");
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    assert!(matches!(p.stream(ws, DEV, &[0, 1, 2], 1, Some(1 << 20)), Err(Error::Invalid(_))), "a ring smaller than the largest region");
    // Nothing mapped: every position streams. Nothing syncs until the end: each read is queued
    // behind a stall, so a copy that reused ring bytes before their release event would
    // overwrite bytes the stalled read has not taken yet.
    let cur = p.stream(ws, DEV, &[0, 1, 2, 3], 3, None).unwrap();
    assert_eq!(cur.home, vec![false; 4]);
    let mut reads = Vec::new();
    for step in 0..3 {
        for r in 0..4u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            s.stall(40);
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
        assert_eq!(cur.passes(), step + 1);
    }
    s.sync();
    let layout = p.layout(ws).unwrap();
    for (r, buf) in &reads {
        check_region(&layout, *r, buf.bytes());
    }
    cur.close().unwrap();
    assert_eq!(p.stats().devices[0].mapped, 0, "closing the cursor returns its ring");
    // One region's ring holds one streamed lease at a time, and refuses rather than hangs.
    let cur = p.stream(ws, DEV, &[1, 2], 1, Some(72 << 20)).unwrap();
    let a = cur.acquire(1, s.raw()).unwrap();
    assert!(cur.acquire(2, s.raw()).is_err(), "an exhausted ring must refuse, not hang");
    a.release(s.raw()).unwrap();
    cur.acquire(2, s.raw()).unwrap().release(s.raw()).unwrap();
    cur.close().unwrap();
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn departures_skip_ahead_or_demand_fill_and_cycles_run_until_close() {
    let fx = Fixture::new("gpu-depart");
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 0, None).unwrap();
    let mut reads = Vec::new();
    let take = |r: u32, reads: &mut Vec<(u32, Pinned)>| {
        let l = cur.acquire(r, s.raw()).unwrap();
        s.stall(5);
        reads.push((r, s.read_async(l.ptr, l.nbytes)));
        l.release(s.raw()).unwrap();
    };
    take(0, &mut reads);
    take(2, &mut reads); // skips position 1: its bytes are handed on unread
    take(0, &mut reads); // the next cycle
    take(0, &mut reads); // again: skips 1 and 2 to the following cycle
    for _ in 0..5 {
        for r in 0..3 {
            take(r, &mut reads);
        }
    }
    s.sync();
    let layout = p.layout(ws).unwrap();
    for (r, buf) in &reads {
        check_region(&layout, *r, buf.bytes());
    }
    assert_eq!(cur.passes(), 8);
    cur.close().unwrap();
    // A region the schedule does not name is a demand acquire (in place, a miss).
    let cur = p.stream(ws, DEV, &[1, 2], 1, None).unwrap();
    let l = cur.acquire(0, s.raw()).unwrap();
    assert!(l.ring_offset.is_none());
    check_region(&layout, 0, &s.read(l.ptr, l.nbytes));
    l.release(s.raw()).unwrap();
    assert_eq!(p.stats().devices[0].counters.misses, 1);
    // Positions past the end are demand acquires too, never an error.
    for r in [1, 2, 1] {
        cur.acquire(r, s.raw()).unwrap().release(s.raw()).unwrap();
    }
    s.sync();
    cur.close().unwrap();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn a_dropped_cursor_closes_itself_once_its_last_lease_is_released() {
    let fx = Fixture::new("gpu-orphan");
    let p = plane();
    let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 0, None).unwrap();
    let l = cur.acquire(0, s.raw()).unwrap();
    drop(cur);
    assert!(p.stats().devices[0].mapped > 0, "a live lease keeps the ring");
    l.release(s.raw()).unwrap();
    s.sync();
    let mut rounds = 0;
    while p.stats().devices[0].mapped != 0 {
        rounds += 1;
        assert!(rounds < 10_000, "an orphaned cursor never closed");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(p.stats().poisoned.is_none());
    p.close().unwrap();
}

/// The driver's access flags for this process at `va`.
fn access(va: u64) -> u64 {
    let d = cuda::driver().unwrap();
    let get = d.mem_get_access.expect("cuMemGetAccess");
    let loc = cuda::MemLocation { kind: cuda::MEM_LOCATION_TYPE_DEVICE, id: DEV };
    let mut flags = 0u64;
    cuda::check("cuMemGetAccess", unsafe { get(&mut flags, &loc, va) }).unwrap();
    flags
}

#[test]
#[ignore = "GPU"]
fn shared_regions_attach_read_only_and_outlive_their_exporter() {
    let fx = Fixture::new("gpu-share");
    let a = plane();
    if !a.device_sharing(DEV).unwrap() {
        eprintln!("device {DEV} cannot share memory: nothing to test");
        return;
    }
    let wa = a.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let s = Stream::new();
    let spans: Vec<u64> = a.layout(wa).unwrap().regions.iter().map(|r| r.span).collect();
    let total: u64 = spans.iter().sum();
    a.want(wa, DEV, None).unwrap().wait().unwrap();
    // A leased region is skipped; the rest become shared and read-only.
    let held = a.acquire(wa, DEV, 2, s.raw()).unwrap();
    let mut exported = a.export_device(wa, DEV, None).unwrap();
    assert_eq!(exported.iter().map(|e| e.region).collect::<Vec<_>>(), vec![0, 1, 3]);
    held.release(s.raw()).unwrap();
    exported.extend(a.export_device(wa, DEV, None).unwrap());
    assert_eq!(exported.len(), 4);
    assert!(a.export_device(wa, DEV, None).unwrap().is_empty(), "shared regions export once");
    assert_eq!(a.stats().devices[0].shared_bytes, total);
    let la = resident(&a, wa, &s, 0);
    assert_eq!(access(la.ptr), u64::from(cuda::MEM_ACCESS_PROT_READ), "the exporter maps it read-only too");
    la.release(s.raw()).unwrap();
    // Another plane attaches the same layout's regions without a copy.
    let b = plane();
    let wb = b.register(&fx.ck, &fx.regions(), PAGE).unwrap();
    let digest = a.layout(wa).unwrap().digest.clone();
    assert_eq!(digest, b.layout(wb).unwrap().digest);
    let first: Vec<(u64, BorrowedFd)> = exported[0].chunks.iter().map(|(n, fd)| (*n, fd.as_fd())).collect();
    assert!(matches!(b.attach_device(wb, DEV, exported[0].region, &first, &"0".repeat(digest.len())), Err(Error::Invalid(_))));
    for e in &exported {
        let chunks: Vec<(u64, BorrowedFd)> = e.chunks.iter().map(|(n, fd)| (*n, fd.as_fd())).collect();
        b.attach_device(wb, DEV, e.region, &chunks, &digest).unwrap();
    }
    assert!(matches!(b.attach_device(wb, DEV, 0, &first, &digest), Err(Error::Invalid(_))));
    // The exporter and every fd go; the attached handles keep the bytes.
    drop(exported);
    a.close().unwrap();
    drop(a);
    for r in 0..4 {
        let l = resident(&b, wb, &s, r);
        assert_eq!(access(l.ptr), u64::from(cuda::MEM_ACCESS_PROT_READ));
        l.release(s.raw()).unwrap();
    }
    let st = b.stats();
    assert_eq!((st.devices[0].mapped, st.devices[0].counters.copies), (0, 0), "attached, never copied");
    assert_eq!(st.devices[0].shared_bytes, total);
    // A drop releases a shared region; the next use fills a private copy.
    assert_eq!(b.drop_regions(wb, DEV, Some(&[1])).unwrap(), spans[1]);
    resident(&b, wb, &s, 1).release(s.raw()).unwrap();
    let st = b.stats();
    assert_eq!((st.devices[0].counters.misses, st.devices[0].mapped), (1, spans[1]));
    s.sync();
    b.close().unwrap();
}

/// Copy failures with copies of the same job still queued on the device (Xid 31: a copy engine
/// wrote a range unmapped after its job failed). A gate holds the job's stream on the device;
/// the fault lands after the job queued at least one copy behind it.
#[cfg(feature = "faults")]
mod copy_faults {
    use super::*;
    use std::time::Duration;
    use varena::faults;

    /// Eight slots: a gated region's five pieces never wait for a slot whose copy the gate holds.
    fn staging_plane() -> Plane {
        open(PlaneConfig { devices: vec![DEV], readers: 4, staging_buffers: 8, staging_bytes: 16 << 20, ..Default::default() })
    }

    fn open_later(g: &faults::Gate) -> std::thread::JoinHandle<()> {
        let open = g.opener();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            open();
        })
    }

    /// The driver page-locks a sealed (read-only) layout: its regions copy by one DMA.
    fn locks_read_only(fx: &Fixture, fd: i32) -> bool {
        let p = plane();
        let ws = p.register(&fx.ck, &fx.regions(), Host::Sealed(fd)).unwrap();
        p.want(ws, DEV, Some(&[0])).unwrap().wait().unwrap();
        let locked = p.stats().io.register_failed == 0;
        p.close().unwrap();
        locked
    }

    /// The device context survived: no copy faulted on an unmapped range.
    fn health() -> Result<(), Error> {
        let (_, ctx) = cuda::primary_context(DEV)?;
        let _g = CtxGuard::enter(ctx)?;
        let d = cuda::driver()?;
        cuda::check("cuCtxSynchronize", unsafe { (d.ctx_synchronize)() })
    }

    /// A sealed region the driver refuses to page-lock is copied through the staging slots,
    /// exact, never as pageable memory, and asked once.
    #[test]
    #[ignore = "GPU"]
    fn a_region_refused_a_lock_is_bounced_and_never_asked_again() {
        let fx = Fixture::new("gpu-refused");
        let layout = varena::Layout::build(fx.ck.header(), &fx.regions()).unwrap();
        let (fd, _fill) = sealed(&fx, &layout, None);
        let p = plane();
        let ws = p.register(&fx.ck, &fx.regions(), Host::Sealed(fd.as_raw_fd())).unwrap();
        let s = Stream::new();
        faults::arm("pin", 1); // the first region's lock is refused
        let mut asked = 0;
        for pass in 0..2 {
            p.want(ws, DEV, None).unwrap().wait().unwrap();
            all_exact(&p, ws, &s);
            if pass == 0 {
                asked = p.stats().io.register_failed;
                assert!(p.drop_regions(ws, DEV, None).unwrap() > 0);
            }
        }
        let st = p.stats();
        let d = &st.devices[0].counters;
        assert!(asked >= 1 && st.io.register_failed == asked, "asked once: {:?}", st.io);
        assert!(d.bounce_copy_bytes > 0, "{d:?}");
        assert_eq!(d.disk_copy_bytes, 0);
        health().unwrap();
        s.sync();
        p.close().unwrap();
    }

    #[test]
    #[ignore = "GPU"]
    fn a_failed_fill_is_reported_only_after_its_queued_copies_drained() {
        let fx = Fixture::new("gpu-drain");
        let layout = varena::Layout::build(fx.ck.header(), &fx.regions()).unwrap();
        let (fd, _fill) = sealed(&fx, &layout, None);
        // (fault point, pass that fails, host): the second staged piece; the record after a DMA.
        for (point, n, host) in [("staged-copy", 2, PAGE), ("host-copy-recorded", 1, Host::Sealed(fd.as_raw_fd()))] {
            if point == "host-copy-recorded" && !locks_read_only(&fx, fd.as_raw_fd()) {
                println!("this driver does not page-lock read-only memory: no DMA to fail");
                continue;
            }
            let p = staging_plane();
            let ws = p.register(&fx.ck, &fx.regions(), host).unwrap();
            let s = Stream::new();
            fx.drop_cache();
            let gate = faults::arm_gate("copy");
            faults::arm(point, n);
            let opener = open_later(&gate);
            let r = p.want(ws, DEV, Some(&[1])).unwrap().wait();
            let drained = gate.is_open();
            // What a runtime does next with a failed region: give its memory back.
            let dropped = p.drop_regions(ws, DEV, Some(&[1]));
            opener.join().unwrap();
            let device = health();
            assert!(r.is_err(), "{point}: the injected failure must surface");
            assert!(drained, "{point}: reported while its copies still waited on the device; device after the unmap: {device:?}");
            dropped.unwrap();
            device.unwrap();
            resident(&p, ws, &s, 1).release(s.raw()).unwrap();
            s.sync();
            assert!(p.stats().poisoned.is_none());
            p.close().unwrap();
            health().unwrap();
        }
    }

    #[test]
    #[ignore = "GPU"]
    fn a_failed_ring_copy_is_reported_only_after_draining_and_the_ring_closes_safely() {
        let fx = Fixture::new("gpu-ringdrain");
        let p = staging_plane();
        let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
        let s = Stream::new();
        fx.drop_cache();
        let gate = faults::arm_gate("copy");
        faults::arm("staged-copy", 2);
        let opener = open_later(&gate);
        let cur = p.stream(ws, DEV, &[1], 1, None).unwrap();
        let r = cur.acquire(1, s.raw());
        let drained = gate.is_open();
        let closed = cur.close(); // unmaps the ring
        opener.join().unwrap();
        let device = health();
        assert!(r.is_err(), "the injected failure must surface at acquire");
        assert!(drained, "a failed ring copy was reported while its pieces still waited on the device; device after the unmap: {device:?}");
        closed.unwrap();
        device.unwrap();
        assert!(p.stats().poisoned.is_none());
        p.close().unwrap();
        health().unwrap();
    }

    #[test]
    #[ignore = "GPU"]
    fn an_undrainable_failure_poisons_and_its_ranges_outlive_the_queued_copies() {
        let fx = Fixture::new("gpu-undrained");
        let p = staging_plane();
        let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
        fx.drop_cache();
        let gate = faults::arm_gate("copy");
        faults::arm("staged-copy", 2);
        faults::arm("drain", 1);
        let r = p.want(ws, DEV, Some(&[1])).unwrap().wait();
        assert!(matches!(r, Err(Error::Poisoned(_))), "{r:?}");
        assert!(matches!(p.close_ws(ws), Err(Error::Poisoned(_))), "a poisoned plane must not unmap");
        assert!(matches!(p.drop_regions(ws, DEV, None), Err(Error::Poisoned(_))));
        // Collecting it waits for the device before anything is unmapped.
        let opener = open_later(&gate);
        drop(p);
        assert!(gate.is_open(), "the plane was torn down while its copies still waited on the device");
        opener.join().unwrap();
        health().unwrap();
    }

    /// A release whose event the driver refuses synchronizes its stream instead; when that fails
    /// too, its bytes are never served again as they are. Either way the lease leaves the books.
    #[test]
    #[ignore = "GPU"]
    fn a_release_without_its_event_still_ends_the_lease() {
        let fx = Fixture::new("gpu-release");
        let p = plane();
        let ws = p.register(&fx.ck, &fx.regions(), PAGE).unwrap();
        let s = Stream::new();
        let layout = p.layout(ws).unwrap();
        let kinds = |p: &Plane| p.events().into_iter().map(|e| e.kind).filter(|k| k.starts_with("release")).collect::<Vec<_>>();
        let leased = |p: &Plane| p.stats().devices[0].leased;
        let misses = |p: &Plane| p.stats().devices[0].counters.misses;
        // In place, synced: the consumer's stalled read is complete when release returns.
        let l = p.acquire(ws, DEV, 1, s.raw()).unwrap();
        s.stall(200);
        let read = s.read_async(l.ptr, l.nbytes);
        faults::arm("release-event", 1);
        l.release(s.raw()).unwrap();
        check_region(&layout, 1, read.bytes());
        assert_eq!(kinds(&p), ["release_synced"]);
        assert_eq!(leased(&p), 0);
        assert!(p.drop_regions(ws, DEV, Some(&[1])).unwrap() > 0, "droppable");
        assert!(matches!(l.release(s.raw()), Err(Error::Invalid(_))), "released twice");
        // Streamed, synced: the bytes are free at once and the ring keeps cycling.
        let cur = p.stream(ws, DEV, &[0, 1, 2], 2, None).unwrap();
        for (i, r) in [0u32, 1, 2, 0, 1, 2].into_iter().enumerate() {
            let l = cur.acquire(r, s.raw()).unwrap();
            check_region(&layout, r, &s.read(l.ptr, l.nbytes));
            if i == 1 {
                faults::arm("release-event", 1);
            }
            l.release(s.raw()).unwrap();
        }
        assert_eq!(kinds(&p), ["release_synced"]);
        cur.close().unwrap();
        // In place, unconfirmed: the next acquire refills it (a miss); it stays droppable.
        let unconfirmed = |l: Lease| {
            faults::arm("release-event", 1);
            faults::arm("release-sync", 1);
            l.release(s.raw()).unwrap();
        };
        unconfirmed(p.acquire(ws, DEV, 1, s.raw()).unwrap());
        assert_eq!(kinds(&p), ["release_unconfirmed"]);
        assert_eq!(leased(&p), 0);
        let before = misses(&p);
        resident(&p, ws, &s, 1).release(s.raw()).unwrap();
        assert_eq!(misses(&p), before + 1, "an unconfirmed region is refilled, not served");
        unconfirmed(p.acquire(ws, DEV, 1, s.raw()).unwrap());
        assert!(p.drop_regions(ws, DEV, Some(&[1])).unwrap() > 0, "droppable");
        // Streamed, unconfirmed: the bytes stay held (a one-region ring has no room for the next),
        // and the cursor still closes.
        let cur = p.stream(ws, DEV, &[0, 2], 1, Some(18 << 20)).unwrap();
        unconfirmed(cur.acquire(0, s.raw()).unwrap());
        assert!(matches!(cur.acquire(2, s.raw()), Err(Error::Invalid(_))), "the held bytes were reused");
        cur.close().unwrap();
        assert_eq!(kinds(&p), ["release_unconfirmed", "release_unconfirmed"]);
        s.sync();
        assert_eq!(p.stats().devices[0].leased, 0);
        assert!(p.stats().poisoned.is_none());
        p.close().unwrap();
        health().unwrap();
    }
}
