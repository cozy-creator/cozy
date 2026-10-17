//! Device paths on a real GPU: fills from the pinned tier and from disk, resident leases,
//! budgets and eviction, the streaming cursor's slot reuse under a slow consumer, driver truth.
//! Ignored by default; run one GPU operator at a time:
//! `flock <locks>/gpu.lock cargo test -p tensorfs-plane --test gpu -- --ignored --test-threads 1`
#![cfg(target_os = "linux")] // The host tier (memfd, seals, OFD locks) is Linux-only.

mod common;

use std::ffi::c_void;
use std::os::fd::{AsFd, BorrowedFd};
use std::sync::atomic::{AtomicU64, Ordering};

use common::*;
use tensorfs_plane::cuda::{self, CUstream, CtxGuard};
use tensorfs_plane::{Error, Lease, Plane, PlaneConfig, Tier, WsId};

const DEV: i32 = 0;
const MIB64: u64 = 64 << 20;

fn plane() -> Plane {
    Plane::open(PlaneConfig {
        devices: vec![DEV],
        readers: 4,
        staging_buffers: 3,
        staging_bytes: 16 << 20,
        ..Default::default()
    })
    .unwrap()
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
        cuda::check("cuMemcpyDtoHAsync", unsafe {
            (d.memcpy_dtoh_async)(v.as_mut_ptr() as *mut c_void, ptr, n as usize, self.0)
        })
        .unwrap();
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

/// Bytes of every part of `region` at `base` (the region's first byte) equal the source.
fn check_region(plane: &Plane, ws: WsId, r: u32, got: &[u8]) {
    let l = plane.layout(ws).unwrap();
    let off = l.regions[r as usize].offset;
    for p in l.parts.iter().filter(|p| p.region == r) {
        let key = p.what.trim_start_matches("unet/").trim_end_matches("#value");
        let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
        let at = (p.offset - off) as usize;
        assert!(got[at..at + n] == bytes_of(key, n)[..], "{} differs on the device", p.what);
    }
}

/// Every host copy was one DMA from locked memory or went through the staging buffers; locked
/// whenever the driver locked the layout. Prints both rates.
fn assert_never_pageable(p: &Plane, what: &str) {
    let st = p.stats();
    let d = &st.devices[0].counters;
    let gbps = |b: u64, ns: u64| b as f64 / ns.max(1) as f64;
    println!(
        "{what}: locked {} B at {:.1} GB/s, bounced {} B at {:.1} GB/s, files {} B; register_failed {}",
        d.host_copy_bytes,
        gbps(d.host_copy_bytes, d.host_copy_ns),
        d.bounce_copy_bytes,
        gbps(d.bounce_copy_bytes, d.bounce_copy_ns),
        d.disk_copy_bytes,
        st.host.counters.register_failed
    );
    if st.host.counters.register_failed == 0 {
        assert_eq!(d.bounce_copy_bytes, 0, "{d:?}");
    }
}

fn resident(plane: &Plane, ws: WsId, s: &Stream, r: u32) -> Lease {
    let l = plane.acquire(ws, DEV, r, s.raw(), None).unwrap();
    let got = s.read(l.ptr, l.nbytes);
    check_region(plane, ws, r, &got);
    l
}

#[test]
#[ignore = "GPU"]
fn fills_from_the_pinned_tier_and_from_disk_are_exact() {
    let fx = Fixture::new("gpu-fill");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    // Pinned budget 0: every device fill reads the store through staging buffers.
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    assert!(st.devices[0].counters.disk_copy_bytes > 70 << 20, "{:?}", st.devices[0]);
    assert_eq!(st.devices[0].counters.host_copy_bytes, 0);
    assert!(p.verify(ws, DEV).unwrap() > 0);

    // Evict, then refill from the pinned tier.
    assert!(p.drop_regions(ws, Tier::Device(DEV), None).unwrap() > 0);
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    assert_eq!(p.stats().host.registered_bytes, p.stats().host.ready_bytes, "pinned tier must be registered");
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    assert!(st.devices[0].counters.host_copy_bytes > 70 << 20, "{:?}", st.devices[0]);
    let gbps = st.devices[0].counters.host_copy_bytes as f64 / st.devices[0].counters.host_copy_ns as f64;
    println!("pinned->device {gbps:.2} GB/s over {} copies", st.devices[0].counters.copies);
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn device_fill_chains_behind_a_running_pinned_fill() {
    let fx = Fixture::new("gpu-chain");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    fx.drop_cache();
    let tickets = p.replicate(ws, &[DEV], None, 0, false).unwrap();
    for t in &tickets {
        t.wait().unwrap();
    }
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    assert_eq!(st.devices[0].counters.disk_copy_bytes, 0, "the device must wait for the pinned fill, not re-read");
    assert!(st.host.direct_bytes > 0);
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn budgets_evict_by_priority_and_leases_block() {
    let fx = Fixture::new("gpu-budget");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    let total: u64 = spans.iter().sum();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    p.set_vram_budget(DEV, total).unwrap();
    p.want(ws, Tier::Device(DEV), Some(&[0]), 5, false).unwrap().wait().unwrap();
    p.want(ws, Tier::Device(DEV), Some(&[1, 2]), 1, false).unwrap().wait().unwrap();

    // A live lease on region 2 blocks it; the budget cut takes region 1 instead.
    let l2 = p.acquire(ws, DEV, 2, s.raw(), None).unwrap();
    let t = p.set_vram_budget(DEV, total - spans[1]).unwrap();
    assert_eq!(t.evicted, vec![("unet".to_string(), 1)]);
    assert!(t.blockers.iter().any(|b| b.starts_with("unet#2")), "{:?}", t.blockers);
    assert!(matches!(p.drop_regions(ws, Tier::Device(DEV), Some(&[2])), Err(Error::LeaseViolation { .. })));
    l2.release(s.raw()).unwrap();

    // A miss refills on demand at the region's last priority, evicting only lower ones.
    p.want(ws, Tier::Device(DEV), Some(&[2]), 0, false).unwrap();
    p.set_vram_budget(DEV, spans[0] + spans[1]).unwrap();
    resident(&p, ws, &s, 1).release(s.raw()).unwrap();
    let st = p.stats();
    assert_eq!(st.devices[0].counters.misses, 1);
    assert_eq!(st.devices[0].counters.evictions, 2, "region 1 by the cut, region 2 by the miss");
    match p.want(ws, Tier::Device(DEV), Some(&[2]), 1, false) {
        Err(Error::Shortfall { pool: "vram", .. }) => {}
        other => panic!("expected Shortfall, got {:?}", other.err()),
    }
    assert!(p.verify(ws, DEV).unwrap() > 0);

    // An unreleased lease poisons the weight set on the device.
    drop(p.acquire(ws, DEV, 0, s.raw(), None).unwrap());
    assert!(matches!(p.acquire(ws, DEV, 0, s.raw(), None), Err(Error::Poisoned(_))));
    s.sync();
}

#[test]
#[ignore = "GPU"]
fn streaming_reuses_a_slot_only_after_its_reader_finished() {
    let fx = Fixture::new("gpu-stream");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    p.set_vram_budget(DEV, (2 * 72) << 20).unwrap();
    assert!(matches!(
        p.stream(ws, DEV, &[0, 1, 2], 2, 1, 0, "tiny", Some(1 << 20)),
        Err(Error::Invalid(_))
    ), "a ring smaller than the largest streamed region is refused");
    // Nothing resident: every position streams through 2 slots. Nothing syncs until the end:
    // each read is queued behind a stall, so a copy that reused its slot before the release
    // event would overwrite bytes the stalled read has not taken yet.
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 3, 0, "denoise", None).unwrap();
    assert_eq!(cur.home, vec![false, false, false]);
    let mut reads = Vec::new();
    for step in 0..3 {
        for r in 0..3u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            s.stall(40);
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
        assert_eq!(cur.passes(), step + 1);
    }
    s.sync();
    let checked = AtomicU64::new(0);
    for (r, buf) in &reads {
        check_region(&p, ws, *r, buf.bytes());
        checked.fetch_add(1, Ordering::Relaxed);
    }
    assert_eq!(checked.load(Ordering::Relaxed), 9);
    cur.close().unwrap();
    let st = p.stats();
    let tag = st.tags.iter().find(|t| t.2 == "denoise").unwrap();
    assert_eq!(tag.3.acquires, 9);
    assert_eq!(tag.3.skipped, 0);
    let per_region: Vec<_> = st.regions.iter().filter(|r| r.2 == "denoise").collect();
    assert_eq!(per_region.len(), 3);
    assert!(per_region.iter().all(|r| r.4.uses == 3 && r.4.compute_ns > 0), "{per_region:?}");
    println!("stream: late {} stall {} ns streamed {} B", tag.3.late, tag.3.stall_ns, tag.3.streamed_bytes);
    assert_eq!(st.devices[0].mapped, 0, "closing the cursor returns its slots");
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn streaming_serves_resident_regions_in_place_and_stays_exact_under_overlap() {
    let fx = Fixture::new("gpu-mixed");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    p.want(ws, Tier::Device(DEV), Some(&[0]), 3, false).unwrap().wait().unwrap();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 1, 4, 0, "mixed", None).unwrap();
    assert_eq!(cur.home, vec![true, false, false]);
    assert!(matches!(p.drop_regions(ws, Tier::Device(DEV), Some(&[0])), Err(Error::LeaseViolation { .. })));
    let mut reads = Vec::new();
    for _ in 0..4 {
        for r in 0..3u32 {
            // Queue every read without syncing: the copies race the reads if the events lie.
            let l = cur.acquire(r, s.raw()).unwrap();
            assert_eq!(l.ring_offset.is_none(), r == 0);
            s.stall(10);
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
    }
    s.sync();
    for (r, buf) in &reads {
        check_region(&p, ws, *r, buf.bytes());
    }
    // Window 1 holds one streamed lease at a time.
    cur.close().unwrap();
    let cur = p.stream(ws, DEV, &[1, 2], 1, 1, 0, "window", None).unwrap();
    let a = cur.acquire(1, s.raw()).unwrap();
    assert!(cur.acquire(2, s.raw()).is_err(), "an exhausted ring must refuse, not hang");
    a.release(s.raw()).unwrap();
    let b = cur.acquire(2, s.raw()).unwrap();
    b.release(s.raw()).unwrap();
    cur.close().unwrap();
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn slabs_back_regions_and_trim_releases_idle_handles() {
    let fx = Fixture::new("gpu-slab");
    let p = plane();
    let ws = fx.register(&p, None);
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    let st = p.stats();
    let spans: u64 = p.layout(ws).unwrap().regions.iter().map(|r| r.span).sum();
    assert_eq!(st.devices[0].mapped, spans);
    assert_eq!(st.devices[0].committed, spans);
    assert!(st.devices[0].granularity <= 2 << 20);
    println!("exportable handles: {}", st.devices[0].exportable);
    p.drop_regions(ws, Tier::Device(DEV), None).unwrap();
    // Dropped handles stay committed-idle (recycled before create) until a budget cut.
    let st = p.stats();
    assert_eq!(st.devices[0].mapped, 0);
    let t = p.set_vram_budget(DEV, MIB64).unwrap();
    assert!(t.freed > 0);
    assert!(p.stats().devices[0].committed <= MIB64);
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn departures_skip_ahead_or_demand_fill_and_cycles_run_until_close() {
    let fx = Fixture::new("gpu-depart");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    // repeat 0: the schedule cycles until close.
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 0, 0, "cfg", None).unwrap();
    let mut reads = Vec::new();
    let take = |r: u32, reads: &mut Vec<(u32, Pinned)>| {
        let l = cur.acquire(r, s.raw()).unwrap();
        s.stall(5);
        reads.push((r, s.read_async(l.ptr, l.nbytes)));
        l.release(s.raw()).unwrap();
    };
    take(0, &mut reads);
    take(2, &mut reads); // skips position 1: its slot is handed on unread
    take(0, &mut reads); // the next cycle
    take(0, &mut reads); // again: skips 1 and 2 to the following cycle
    // The loop starts at 0 while the schedule is at 1: two more skips.
    for _ in 0..5 {
        for r in 0..3 {
            take(r, &mut reads);
        }
    }
    s.sync();
    for (r, buf) in &reads {
        check_region(&p, ws, *r, buf.bytes());
    }
    assert_eq!(cur.passes(), 8);
    cur.close().unwrap();
    let st = p.stats();
    let tag = st.tags.iter().find(|t| t.2 == "cfg").unwrap();
    assert_eq!(tag.3.skipped, 5);
    // A region the schedule does not name is a demand acquire (resident, a miss).
    let cur = p.stream(ws, DEV, &[1, 2], 1, 1, 0, "narrow", None).unwrap();
    let l = cur.acquire(0, s.raw()).unwrap();
    assert!(l.ring_offset.is_none());
    let got = s.read(l.ptr, l.nbytes);
    check_region(&p, ws, 0, &got);
    l.release(s.raw()).unwrap();
    assert!(p.stats().devices[0].counters.misses >= 1);
    // Positions past the end are demand acquires too, never an error.
    for r in [1, 2, 1] {
        let l = cur.acquire(r, s.raw()).unwrap();
        l.release(s.raw()).unwrap();
    }
    s.sync();
    cur.close().unwrap();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn a_dropped_cursor_closes_itself_and_floors_refuse_early() {
    let fx = Fixture::new("gpu-orphan");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    // A ring for the 70 MiB region does not fit a 64 MiB budget at all: nothing could help.
    p.set_vram_budget(DEV, MIB64).unwrap();
    match p.stream(ws, DEV, &[0, 1, 2], 1, 1, 0, "floor", None) {
        Err(Error::BelowFloor { pool: "vram", .. }) => {}
        other => panic!("expected BelowFloor, got {:?}", other.err()),
    }
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 0, 0, "orphan", None).unwrap();
    let l = cur.acquire(0, s.raw()).unwrap();
    drop(cur);
    assert!(p.stats().devices[0].mapped > 0, "a live lease keeps the slots");
    l.release(s.raw()).unwrap();
    s.sync();
    // The last copy in flight reaps the cursor on the mover thread.
    let mut rounds = 0;
    while p.stats().devices[0].mapped != 0 {
        rounds += 1;
        assert!(rounds < 10_000, "orphaned cursor never closed");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(p.stats().poisoned.is_none());
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn cutting_just_wanted_regions_waits_for_every_staged_chunk() {
    // W3 R24: want cold from disk through staging (pinned budget 0; 16 MiB staging puts the
    // 70 MiB region in 5 chunks), acquire and release each region without the consumer ever
    // waiting, then cut the budget to zero before anything reads them. A copy still writing an
    // unmapped range would surface as ILLEGAL_ADDRESS at the next driver call.
    let fx = Fixture::new("gpu-r24");
    for _ in 0..4 {
        let p = plane();
        let ws = fx.register(&p, None);
        let s = Stream::new();
        p.set_vram_budget(DEV, 1 << 30).unwrap();
        fx.drop_cache();
        let _ticket = p.want(ws, Tier::Device(DEV), None, 5, false).unwrap();
        for r in 0..3 {
            p.acquire(ws, DEV, r, s.raw(), None).unwrap().release(s.raw()).unwrap();
        }
        let t = p.set_vram_budget(DEV, 0).unwrap();
        assert_eq!(t.evicted.len(), 3, "{t:?}");
        let ev = {
            let d = cuda::driver().unwrap();
            let mut e = std::ptr::null_mut();
            cuda::check("cuEventCreate", unsafe { (d.event_create)(&mut e, cuda::EVENT_DEFAULT) }).unwrap();
            cuda::check("cuEventRecord", unsafe { (d.event_record)(e, s.0) }).unwrap();
            e
        };
        s.sync();
        unsafe { (cuda::driver().unwrap().event_destroy)(ev) };
        p.set_vram_budget(DEV, 1 << 30).unwrap();
        for r in 0..3 {
            resident(&p, ws, &s, r).release(s.raw()).unwrap();
        }
        assert!(p.stats().poisoned.is_none());
        s.sync();
        p.close().unwrap();
    }
}

/// W3 G4: a repeat run differed only where streamed copies came disk → staging → ring.
/// Stream cyclically while the source of each copy changes under it: a Worker plane sharing the
/// memfd lets go of it (its eviction only ends its claim: ours keep the bytes), our own pinned
/// budget drops to zero with a cold page cache (every copy staged from disk), then the tier
/// refills while streaming, then shrinks to one region. Every streamed read is checked against
/// the source.
fn mixed_source_stream(copy_streams: usize) {
    let fx = Fixture::new("gpu-mixsrc");
    let worker = host_only();
    let wws = fx.register(&worker, None);
    worker.set_pinned_budget(1 << 30).unwrap();
    worker.want(wws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    let p = Plane::open(PlaneConfig {
        devices: vec![DEV],
        readers: 4,
        copy_streams,
        staging_buffers: 3,
        staging_bytes: 16 << 20,
        ..Default::default()
    })
    .unwrap();
    let ws = fx.register(&p, Some(worker.host_fd(wws).unwrap()));
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    let s = Stream::new();
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 0, 0, "mix", None).unwrap();
    let mut bad = Vec::new();
    for pass in 0..28u32 {
        match pass {
            4 => drop(worker.set_pinned_budget(0).unwrap()),
            8..=10 => {
                p.set_pinned_budget(0).unwrap();
                fx.drop_cache();
            }
            16 => {
                p.set_pinned_budget(1 << 30).unwrap();
                p.want(ws, Tier::Pinned, None, 0, false).unwrap();
            }
            22 | 23 => drop(p.set_pinned_budget(spans[0]).unwrap()),
            _ => {}
        }
        let mut reads = Vec::new();
        for r in 0..3u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            if pass % 5 == 0 {
                bad.extend(l.verify_bytes().unwrap().into_iter().map(|w| format!("pass {pass} verify {w}")));
            }
            s.stall(2);
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
        s.sync();
        let l = p.layout(ws).unwrap();
        for (r, buf) in &reads {
            let off = l.regions[*r as usize].offset;
            for part in l.parts.iter().filter(|x| x.region == *r) {
                let key = part.what.trim_start_matches("unet/").trim_end_matches("#value");
                let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
                let at = (part.offset - off) as usize;
                if buf.bytes()[at..at + n] != bytes_of(key, n)[..] {
                    bad.push(format!("pass {pass} {}", part.what));
                }
            }
        }
    }
    let st = p.stats();
    let d = &st.devices[0].counters;
    println!(
        "copy_streams {copy_streams}: host copies {} B, disk-staged {} B, mismatches {}",
        d.host_copy_bytes, d.disk_copy_bytes, bad.len()
    );
    assert!(bad.is_empty(), "streamed bytes differ from the source: {bad:?}");
    assert!(d.host_copy_bytes > 0 && d.disk_copy_bytes > 0, "both sources must be exercised: {d:?}");
    cur.close().unwrap();
    s.sync();
    p.close().unwrap();
    worker.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn streaming_stays_exact_across_pinned_disk_and_foreign_punch() {
    mixed_source_stream(1);
    mixed_source_stream(2);
}

#[test]
#[ignore = "GPU"]
fn verify_bytes_refuses_once_the_lease_is_released() {
    let fx = Fixture::new("gpu-verify");
    let p = plane();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    let l = resident(&p, ws, &s, 0);
    assert!(l.verify_bytes().unwrap().is_empty());
    l.release(s.raw()).unwrap();
    // Released bytes may be evicted at once: reading them is a read of an unmapped range.
    assert!(matches!(l.verify_bytes(), Err(Error::Invalid(_))));
    s.sync();
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
fn shared_regions_attach_read_only_outlive_their_exporter_and_hold_no_budget() {
    let fx = Fixture::new("gpu-share");
    let a = plane();
    if !a.device_sharing(DEV).unwrap() {
        eprintln!("device {DEV} cannot share memory: nothing to test");
        return;
    }
    let wa = fx.register(&a, None);
    let s = Stream::new();
    let spans: Vec<u64> = a.layout(wa).unwrap().regions.iter().map(|r| r.span).collect();
    let total: u64 = spans.iter().sum();
    a.set_vram_budget(DEV, 512 << 20).unwrap();
    a.want(wa, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    let before = a.stats().devices[0].committed;

    // A leased region is skipped; the rest leave the budget and become read-only.
    let held = a.acquire(wa, DEV, 2, s.raw(), None).unwrap();
    let first = a.export_device(wa, DEV, None).unwrap();
    assert_eq!(first.iter().map(|e| e.region).collect::<Vec<_>>(), vec![0, 1]);
    held.release(s.raw()).unwrap();
    let mut exported = first;
    exported.extend(a.export_device(wa, DEV, None).unwrap());
    assert_eq!(exported.len(), 3);
    assert!(a.export_device(wa, DEV, None).unwrap().is_empty(), "shared regions export once");
    let st = a.stats();
    assert_eq!(st.devices[0].committed, before - total);
    assert_eq!(st.devices[0].shared_bytes, total);
    let la = resident(&a, wa, &s, 0);
    assert_eq!(access(la.ptr), u64::from(cuda::MEM_ACCESS_PROT_READ), "the exporter maps it read-only too");
    la.release(s.raw()).unwrap();

    // Another plane attaches the same layout's regions without a copy.
    let b = plane();
    let wb = fx.register(&b, None);
    assert_eq!(a.layout(wa).unwrap().digest, b.layout(wb).unwrap().digest);
    b.set_vram_budget(DEV, 512 << 20).unwrap();
    let digest = a.layout(wa).unwrap().digest.clone();
    // A holder that names another layout is refused before any fd is imported.
    let first: Vec<(u64, BorrowedFd)> = exported[0].chunks.iter().map(|(n, fd)| (*n, fd.as_fd())).collect();
    let other = "0".repeat(digest.len());
    assert!(matches!(b.attach_device(wb, DEV, exported[0].region, &first, &other), Err(Error::Invalid(_))));
    assert_eq!(b.stats().devices[0].shared_bytes, 0);
    for e in &exported {
        let chunks: Vec<(u64, BorrowedFd)> = e.chunks.iter().map(|(n, fd)| (*n, fd.as_fd())).collect();
        b.attach_device(wb, DEV, e.region, &chunks, &digest).unwrap();
    }
    assert!(matches!(b.attach_device(wb, DEV, 0, &first, &digest), Err(Error::Invalid(_))));
    let short: Vec<(u64, BorrowedFd)> = exported[1].chunks.iter().map(|(n, fd)| (*n / 2, fd.as_fd())).collect();
    assert!(matches!(b.attach_device(wb, DEV, 2, &short, &digest), Err(Error::Invalid(_))));

    // The exporter and every fd go; the attached handles keep the bytes.
    drop(exported);
    a.close().unwrap();
    drop(a);
    for r in 0..3 {
        let l = resident(&b, wb, &s, r);
        assert_eq!(access(l.ptr), u64::from(cuda::MEM_ACCESS_PROT_READ));
        l.release(s.raw()).unwrap();
    }
    let st = b.stats();
    assert_eq!((st.devices[0].committed, st.devices[0].counters.copies), (0, 0), "attached, never copied");
    assert_eq!(st.devices[0].shared_bytes, total);
    assert_eq!(st.sets[0].shared, vec![(DEV, st.sets[0].devices[0].1)]);
    assert!(b.verify(wb, DEV).unwrap() >= 3);

    // A budget of 0 evicts nothing shared; a drop releases it and the next use fills privately.
    assert!(b.set_vram_budget(DEV, 0).unwrap().evicted.is_empty());
    assert_eq!(b.drop_regions(wb, Tier::Device(DEV), Some(&[1])).unwrap(), spans[1]);
    b.set_vram_budget(DEV, 512 << 20).unwrap();
    resident(&b, wb, &s, 1).release(s.raw()).unwrap();
    let st = b.stats();
    assert_eq!(st.devices[0].counters.misses, 1);
    assert_eq!(st.devices[0].committed, spans[1]);
    assert_eq!(st.devices[0].shared_bytes, total - spans[1]);
    s.sync();
    b.close().unwrap();
}

/// Copy failures with copies of the same job still queued on the device (Xid 31: a copy engine
/// wrote a range unmapped after its job failed). A gate holds the job's stream on the device;
/// the fault lands after the job enqueued at least one copy behind it.
#[cfg(feature = "faults")]
mod copy_faults {
    use super::*;
    use std::time::Duration;
    use tensorfs_plane::faults;

    fn staging_plane() -> Plane {
        // Eight bounces: the staged region's five chunks never wait for a bounce to come back.
        Plane::open(PlaneConfig {
            devices: vec![DEV],
            readers: 4,
            staging_buffers: 8,
            staging_bytes: 16 << 20,
            ..Default::default()
        })
        .unwrap()
    }

    fn open_later(g: &faults::Gate) -> std::thread::JoinHandle<()> {
        let open = g.opener();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400));
            open();
        })
    }

    /// The device context survived: no copy faulted on an unmapped range.
    fn health() -> Result<(), Error> {
        let (_, ctx) = cuda::primary_context(DEV)?;
        let _g = CtxGuard::enter(ctx)?;
        let d = cuda::driver()?;
        cuda::check("cuCtxSynchronize", unsafe { (d.ctx_synchronize)() })
    }

    fn healthy() {
        health().unwrap();
    }

    /// A region the driver refuses to page-lock is copied through the staging buffers, exact,
    /// never as pageable memory, and asked once.
    #[test]
    #[ignore = "GPU"]
    fn a_region_refused_a_lock_is_bounced_and_never_asked_again() {
        use std::os::fd::AsRawFd;
        let fx = Fixture::new("gpu-sealed-refused");
        let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
        let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
        let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
        let fd = tensorfs_plane::host::fill_sealed("unet", &layout, &source, 3, &tensorfs_plane::io::IoTally::default(), None).unwrap();
        drop(source);
        let p = staging_plane();
        let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
        let s = Stream::new();
        p.set_pinned_budget(0).unwrap();
        p.set_vram_budget(DEV, 512 << 20).unwrap();
        faults::arm("pin", 1); // the first region's lock is refused
        let mut asked = 0;
        for pass in 0..2 {
            p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
            for r in 0..3 {
                resident(&p, ws, &s, r).release(s.raw()).unwrap();
            }
            if pass == 0 {
                asked = p.stats().host.counters.register_failed;
                assert!(p.drop_regions(ws, Tier::Device(DEV), None).unwrap() > 0);
            }
        }
        let st = p.stats();
        let d = &st.devices[0].counters;
        assert!(asked >= 1 && st.host.counters.register_failed == asked, "asked once: {:?}", st.host.counters);
        assert!(d.bounce_copy_bytes > 0, "{d:?}");
        assert_eq!(d.disk_copy_bytes, 0);
        healthy();
        s.sync();
        p.close().unwrap();
    }

    #[test]
    #[ignore = "GPU"]
    fn a_failed_fill_is_reported_only_after_its_queued_copies_drained() {
        let fx = Fixture::new("gpu-drain");
        // (fault point, pass that fails, source): the second staged chunk; the record after a
        // pinned-tier copy.
        for (point, n, pinned) in [("staged-copy", 2, false), ("host-copy-recorded", 1, true)] {
            let p = staging_plane();
            let ws = fx.register(&p, None);
            let s = Stream::new();
            p.set_vram_budget(DEV, 1 << 30).unwrap();
            if pinned {
                p.set_pinned_budget(1 << 30).unwrap();
                p.want(ws, Tier::Pinned, Some(&[1]), 0, false).unwrap().wait().unwrap();
            } else {
                fx.drop_cache();
            }
            let gate = faults::arm_gate("copy");
            faults::arm(point, n);
            let opener = open_later(&gate);
            let r = p.want(ws, Tier::Device(DEV), Some(&[1]), 0, false).unwrap().wait();
            let drained = gate.is_open();
            // What a runtime does next with a failed region: give its memory back.
            let dropped = p.drop_regions(ws, Tier::Device(DEV), Some(&[1]));
            opener.join().unwrap();
            let device = health();
            assert!(r.is_err(), "{point}: the injected failure must surface");
            assert!(drained, "{point}: the failure was reported while its copies still waited on the device; device after the unmap: {device:?}");
            dropped.unwrap();
            device.unwrap();
            // The region refills exactly.
            resident(&p, ws, &s, 1).release(s.raw()).unwrap();
            s.sync();
            assert!(p.stats().poisoned.is_none());
            p.close().unwrap();
            healthy();
        }
    }

    #[test]
    #[ignore = "GPU"]
    fn a_failed_ring_copy_is_reported_only_after_draining_and_the_ring_closes_safely() {
        let fx = Fixture::new("gpu-ringdrain");
        let p = staging_plane();
        let ws = fx.register(&p, None);
        let s = Stream::new();
        p.set_vram_budget(DEV, 1 << 30).unwrap();
        fx.drop_cache();
        let gate = faults::arm_gate("copy");
        faults::arm("staged-copy", 2);
        let opener = open_later(&gate);
        let cur = p.stream(ws, DEV, &[1], 1, 1, 0, "ring", None).unwrap();
        let r = cur.acquire(1, s.raw());
        let drained = gate.is_open();
        let closed = cur.close(); // unmaps the ring
        opener.join().unwrap();
        let device = health();
        assert!(r.is_err(), "the injected failure must surface at acquire");
        assert!(drained, "a failed ring copy was reported while its chunks still waited on the device; device after the unmap: {device:?}");
        closed.unwrap();
        device.unwrap();
        assert!(p.stats().poisoned.is_none());
        p.close().unwrap();
        healthy();
    }

    #[test]
    #[ignore = "GPU"]
    fn an_undrainable_failure_poisons_and_its_ranges_outlive_the_queued_copies() {
        let fx = Fixture::new("gpu-undrained");
        let p = staging_plane();
        let ws = fx.register(&p, None);
        p.set_vram_budget(DEV, 1 << 30).unwrap();
        fx.drop_cache();
        let gate = faults::arm_gate("copy");
        faults::arm("staged-copy", 2);
        faults::arm("drain", 1);
        let r = p.want(ws, Tier::Device(DEV), Some(&[1]), 0, false).unwrap().wait();
        assert!(matches!(r, Err(Error::Poisoned(_))), "{r:?}");
        assert!(matches!(p.close_ws(ws), Err(Error::Poisoned(_))), "a poisoned plane must not unmap");
        assert!(matches!(p.drop_regions(ws, Tier::Device(DEV), None), Err(Error::Poisoned(_))));
        // Collecting it waits for the device before anything is unmapped: the copies still
        // queued behind the gate land in mapped memory.
        let opener = open_later(&gate);
        drop(p);
        assert!(gate.is_open(), "the plane was torn down while its copies still waited on the device");
        opener.join().unwrap();
        healthy();
    }

    /// Ordering, not a fault reproduction: the driver (580) already makes cuMemHostUnregister
    /// wait for queued DMA from the range, so this passes without the plane's own wait too.
    #[test]
    #[ignore = "GPU"]
    fn dropping_the_plane_under_a_queued_pinned_copy_keeps_its_range_until_the_copy_ran() {
        let fx = Fixture::new("gpu-dropq");
        let p = staging_plane();
        let ws = fx.register(&p, None);
        p.set_vram_budget(DEV, 1 << 30).unwrap();
        p.set_pinned_budget(1 << 30).unwrap();
        p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
        let kept = unsafe { libc::dup(p.host_fd(ws).unwrap()) };
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { libc::fstat(kept, &mut st) }, 0);
        let mapped = || {
            std::fs::read_to_string("/proc/self/maps")
                .unwrap()
                .lines()
                .any(|l| l.contains("memfd:tfs-plane") && l.split_whitespace().nth(4) == Some(&st.st_ino.to_string()))
        };
        assert!(mapped());
        let gate = faults::arm_gate("copy");
        drop(p.want(ws, Tier::Device(DEV), Some(&[1]), 0, false).unwrap());
        // The queued copy's completion now holds the plane's last reference.
        drop(p);
        std::thread::sleep(Duration::from_millis(300));
        let still = mapped();
        gate.open();
        assert!(still, "the pinned range was unmapped while a copy from it waited on the device");
        let mut rounds = 0;
        while mapped() {
            rounds += 1;
            assert!(rounds < 5000, "the plane never finished dropping");
            std::thread::sleep(Duration::from_millis(1));
        }
        healthy();
        unsafe { libc::close(kept) };
    }
}

#[test]
#[ignore = "GPU"]
fn a_sealed_layout_reaches_the_device_pinned_inside_the_budget_and_locked_past_it() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-sealed");
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
    let tally = tensorfs_plane::io::IoTally::default();
    let fd = tensorfs_plane::host::fill_sealed("unet", &layout, &source, 3, &tally, None).unwrap();
    drop(source);
    let p = plane();
    let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let s = Stream::new();
    // Region 0 inside the pinned budget (registered read-only for DMA); 1 and 2 past it.
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    p.set_pinned_budget(spans[0]).unwrap();
    p.want(ws, Tier::Pinned, Some(&[0]), 0, false).unwrap().wait().unwrap();
    let st = p.stats();
    println!("sealed: registered {} of ready {}; register_failed {}", st.host.registered_bytes, st.host.ready_bytes, st.host.counters.register_failed);
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    let d = &st.devices[0].counters;
    assert_eq!(d.disk_copy_bytes, 0, "a sealed layout never reads the store");
    assert!(d.host_copy_bytes + d.bounce_copy_bytes > 70 << 20, "{d:?}");
    assert_eq!(st.host.counters.fill_bytes, 0);
    // Past the budget: locked on its first copy, or bounced where the driver refuses.
    assert_never_pageable(&p, "sealed, pinned budget for region 0");
    // Streaming past the budget copies the unpinned regions straight from the layout.
    assert!(p.drop_regions(ws, Tier::Device(DEV), None).unwrap() > 0);
    p.set_vram_budget(DEV, (2 * 72) << 20).unwrap();
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 2, 0, "denoise", None).unwrap();
    for _ in 0..2 {
        for r in 0..3u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            check_region(&p, ws, r, &s.read(l.ptr, l.nbytes));
            l.release(s.raw()).unwrap();
        }
    }
    cur.close().unwrap();
    s.sync();
    p.close().unwrap();
}

#[test]
#[ignore = "GPU"]
fn a_layout_still_filling_reaches_the_device_region_by_region() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-progressive");
    let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, fill) = tensorfs_plane::host::open_sealed("unet", &layout, None, None).unwrap();
    let p = plane();
    let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let s = Stream::new();
    // Region 0 in the pinned books (waits, then registers); 1 and 2 locked on their first copy
    // once Ready (the mover waits for that).
    let spans: Vec<u64> = p.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    p.set_pinned_budget(spans[0]).unwrap();
    let pinned = p.want(ws, Tier::Pinned, Some(&[0]), 0, false).unwrap();
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    let device = p.want(ws, Tier::Device(DEV), None, 0, false).unwrap();
    assert!(!device.done().unwrap(), "no device copy before its region is Ready");
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let filler = std::thread::spawn(move || fill.run(&layout, &source, 3, &tensorfs_plane::io::IoTally::default()));
    pinned.wait().unwrap();
    device.wait().unwrap();
    filler.join().unwrap().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    assert_eq!(p.stats().devices[0].counters.disk_copy_bytes, 0);
    s.sync();
    p.close().unwrap();
}

/// F's R1 cells (RTX 3070, 3 GiB pool): a sealed region outside the executor's pinned books was
/// copied as pageable memory, about 2 GB/s against 11 through the staging buffers, and every
/// request after the machine staged a model's streamed regions was 1.7x slower than the first.
/// Each such region is page-locked on its first copy, with no pinned budget charged.
#[test]
#[ignore = "GPU"]
fn a_sealed_region_outside_the_pinned_books_is_locked_on_its_first_copy() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-sealed-locked");
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
    let fd = tensorfs_plane::host::fill_sealed("unet", &layout, &source, 3, &tensorfs_plane::io::IoTally::default(), None).unwrap();
    drop(source);
    let p = plane();
    let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let s = Stream::new();
    p.set_pinned_budget(0).unwrap();
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    for pass in 0..2 {
        p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
        for r in 0..3 {
            resident(&p, ws, &s, r).release(s.raw()).unwrap();
        }
        if pass == 0 {
            assert!(p.drop_regions(ws, Tier::Device(DEV), None).unwrap() > 0);
        }
    }
    let st = p.stats();
    let d = &st.devices[0].counters;
    assert_eq!(st.host.used, 0, "no pinned budget was charged");
    assert_eq!(d.disk_copy_bytes, 0);
    assert!(d.host_copy_bytes + d.bounce_copy_bytes >= (2 * 70) << 20, "{d:?}");
    assert!(st.host.counters.register_failed <= 1, "a refusal holds for the layout: {:?}", st.host.counters);
    assert_never_pageable(&p, "sealed, outside the pinned books");
    s.sync();
    p.close().unwrap();
}

/// A streamed layout's machine on a thread: its window serves until the returned flag is set.
fn served_window(fx: &Fixture, slots: usize) -> (std::os::fd::OwnedFd, std::sync::Arc<std::sync::atomic::AtomicBool>, std::thread::JoinHandle<tensorfs_plane::Result<()>>) {
    let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, window) = tensorfs_plane::window::open_window("unet", &layout, slots).unwrap();
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = stop.clone();
    let t = std::thread::spawn(move || window.serve(&layout, &source, 3, &tensorfs_plane::io::IoTally::default(), &s));
    (fd, stop, t)
}

#[test]
#[ignore = "GPU"]
fn a_streamed_layout_reaches_the_device_through_one_slot() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-window");
    fx.drop_cache();
    let (fd, stop, machine) = served_window(&fx, 1);
    let p = plane();
    let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    assert!(matches!(p.want(ws, Tier::Pinned, None, 0, false), Err(Error::BelowFloor { .. })));
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    assert_eq!(st.devices[0].counters.disk_copy_bytes, 0, "the executor reads no store");
    // A slot is refilled with another region once its copy lets go: never locked, bounced.
    assert!(st.devices[0].counters.bounce_copy_bytes > 70 << 20, "{:?}", st.devices[0]);
    assert_eq!(st.devices[0].counters.host_copy_bytes, 0);
    s.sync();
    p.close().unwrap();
    stop.store(true, Ordering::Release);
    machine.join().unwrap().unwrap();
}

#[test]
#[ignore = "GPU"]
fn a_ring_streams_a_streamed_layout_pass_after_pass() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-window-ring");
    let (fd, stop, machine) = served_window(&fx, 2);
    let p = plane();
    let ws = p.register_sealed("unet", &fx.plan(), &regions(), fd.as_raw_fd()).unwrap();
    let s = Stream::new();
    p.set_vram_budget(DEV, (2 * 72) << 20).unwrap();
    // Neither the host (two slots) nor the device (two ring slots) holds the weight set: every
    // position comes from disk through the machine's window, three passes over.
    let cur = p.stream(ws, DEV, &[0, 1, 2], 2, 3, 0, "denoise", None).unwrap();
    let mut reads = Vec::new();
    for _ in 0..3 {
        for r in 0..3u32 {
            let l = cur.acquire(r, s.raw()).unwrap();
            reads.push((r, s.read_async(l.ptr, l.nbytes)));
            l.release(s.raw()).unwrap();
        }
    }
    s.sync();
    for (r, buf) in &reads {
        check_region(&p, ws, *r, buf.bytes());
    }
    assert_eq!(reads.len(), 9);
    cur.close().unwrap();
    p.close().unwrap();
    stop.store(true, Ordering::Release);
    machine.join().unwrap().unwrap();
}

#[test]
#[ignore = "GPU"]
fn holes_reach_the_device_from_the_object_files_and_staged_regions_from_the_layout() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new("gpu-staged");
    fx.drop_cache();
    let layout = tensorfs_plane::layout::Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, fill) = tensorfs_plane::host::open_sealed("unet", &layout, None, Some(&[1])).unwrap();
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    fill.fill(&layout, &source, 3, &tensorfs_plane::io::IoTally::default()).unwrap();
    let files = fx.objects.iter().map(|o| (o.clone(), std::fs::File::open(fx.store.blob_path(&o.sha256)).unwrap())).collect();
    let p = plane();
    let ws = p.register_staged("unet", &fx.plan(), &regions(), fd.as_raw_fd(), files).unwrap();
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap(); // region 1 only
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    let st = p.stats();
    assert!(st.devices[0].counters.disk_copy_bytes > 3 << 20, "holes 0 and 2 came from the files: {:?}", st.devices[0]);
    let d = &st.devices[0].counters;
    assert!(d.host_copy_bytes + d.bounce_copy_bytes >= 70 << 20, "region 1 from the layout: {d:?}");
    s.sync();
    p.close().unwrap();
    drop(fill);
}

/// Run 5894's shape, a managed executor's sealed layout whose holes read the object files it
/// was handed: a stage's region is unmapped mid-stage (out-of-memory recovery), and its next
/// demand fill fails. That acquire reports the fill's own cause, and a later acquire fills the
/// region again: one failed fill never fails every acquire for the rest of the process.
#[test]
#[ignore = "GPU"]
fn a_failed_demand_fill_reports_its_cause_and_the_next_acquire_fills_again() {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;
    use tensorfs_plane::layout::{ItemSource, Layout};
    let fx = Fixture::new("gpu-refill");
    let layout = Layout::build(&fx.plan(), &regions()).unwrap();
    let (fd, fill) = tensorfs_plane::host::open_sealed("unet", &layout, None, Some(&[1])).unwrap();
    let (lease, _) = tensorfs_core::read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let source = tensorfs_plane::io::Source::new(fx.store.clone(), fx.meta.clone(), lease, true, 0);
    fill.fill(&layout, &source, 3, &tensorfs_plane::io::IoTally::default()).unwrap();
    // The handed files are copies, so the hole region 2 reads can be made to fail and recover.
    let dir = fx.root.join("handed");
    std::fs::create_dir_all(&dir).unwrap();
    let copy = |o: &tensorfs_core::ids::ObjectRef| {
        let path = dir.join(&o.sha256);
        std::fs::copy(fx.store.blob_path(&o.sha256), &path).unwrap();
        (o.clone(), std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap())
    };
    let files: Vec<_> = fx.objects.iter().map(copy).collect();
    let hole = layout.regions[2]
        .items
        .iter()
        .find_map(|it| match &it.source {
            ItemSource::Object(o) => Some(o.obj.sha256.clone()),
            ItemSource::Inline(_) => None,
        })
        .unwrap();
    let file = files.iter().find(|(o, _)| o.sha256 == hole).unwrap().1.try_clone().unwrap();
    let whole = std::fs::read(fx.store.blob_path(&hole)).unwrap();
    let p = plane();
    let ws = p.register_staged("unet", &fx.plan(), &regions(), fd.as_raw_fd(), files).unwrap();
    let s = Stream::new();
    p.set_vram_budget(DEV, 512 << 20).unwrap();
    p.want(ws, Tier::Device(DEV), None, 0, false).unwrap().wait().unwrap();
    for r in 0..3 {
        resident(&p, ws, &s, r).release(s.raw()).unwrap();
    }
    s.sync();
    // Mid-stage, room is made: region 2 leaves the device. Its object then reads short.
    assert!(p.drop_regions(ws, Tier::Device(DEV), Some(&[2])).unwrap() > 0);
    file.set_len(whole.len() as u64 / 2).unwrap();
    let failed = p.acquire(ws, DEV, 2, s.raw(), None).err();
    assert!(matches!(failed, Some(Error::Io(_))), "the fill's own cause: {failed:?}");
    assert!(p.events().iter().any(|e| e.kind == "fill_failed" && e.region == Some(2)));
    // The object reads whole again: the next acquire fills the region, exact.
    file.write_all_at(&whole[whole.len() / 2..], whole.len() as u64 / 2).unwrap();
    resident(&p, ws, &s, 2).release(s.raw()).unwrap();
    s.sync();
    assert!(p.stats().poisoned.is_none());
    p.close().unwrap();
    drop(fill);
}
