//! Device paths on a real GPU: fills from the pinned tier and from disk, resident leases,
//! budgets and eviction, the streaming cursor's slot reuse under a slow consumer, driver truth.
//! Ignored by default; run one GPU operator at a time:
//! `flock <locks>/gpu.lock cargo test -p tensorfs-plane --test gpu -- --ignored --test-threads 1`

mod common;

use std::ffi::c_void;
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
    fn an_undrainable_failure_poisons_and_nothing_is_unmapped_under_it() {
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
        drop(p);
        gate.open();
        std::thread::sleep(Duration::from_millis(200));
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
