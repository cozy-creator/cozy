//! What a plane leaves behind in its process: descriptors, plane memfds, store files and
//! threads return to the baseline however a plane ends: closed, or collected unclosed with a
//! handle outliving it. Alone in its binary: the census is process-wide.

mod common;

use std::time::Duration;

use common::*;
use tensorfs_plane::cuda::{self, CUstream, CtxGuard};
use tensorfs_plane::{Error, Plane, PlaneConfig, Tier};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Census {
    fds: usize,
    memfds: usize,
    store: usize,
    threads: usize,
}

fn census(fx: &Fixture) -> Census {
    let root = fx.root.to_string_lossy().to_string();
    let links: Vec<String> = std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|e| std::fs::read_link(e.ok()?.path()).ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let threads = std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|e| std::fs::read_to_string(e.ok()?.path().join("comm")).ok())
        .filter(|name| name.starts_with("tfs-plane"))
        .count();
    Census {
        // The directory read holds one descriptor of its own.
        fds: links.len() - 1,
        memfds: links.iter().filter(|l| l.contains("memfd:tfs-plane")).count(),
        store: links.iter().filter(|l| l.starts_with(&root)).count(),
        threads,
    }
}

/// The baseline: the census once no plane thread is left (a joined thread's task entry can
/// outlive the join by a moment).
fn baseline(fx: &Fixture) -> Census {
    let quiet = Census { threads: 0, ..census(fx) };
    let seen = settled(fx, &quiet);
    assert_eq!(seen.threads, 0, "plane threads outlived every plane");
    seen
}

/// Threads of an unclosed plane exit on their own: wait for the census to settle.
fn settled(fx: &Fixture, want: &Census) -> Census {
    let mut seen = census(fx);
    for _ in 0..2000 {
        if &seen == want {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
        seen = census(fx);
    }
    seen
}

#[test]
fn host_planes_leave_no_descriptors_or_threads() {
    let fx = Fixture::new("census");
    let cycle = |closed: bool| {
        let plane = host_only();
        let a = fx.register(&plane, None);
        let b = fx.register(&plane, Some(plane.host_fd(a).unwrap()));
        plane.set_pinned_budget(1 << 30).unwrap();
        let ticket = plane.want(a, Tier::Pinned, None, 0, false).unwrap();
        ticket.wait().unwrap();
        plane.want(b, Tier::Pinned, Some(&[0]), 0, false).unwrap().wait().unwrap();
        if closed {
            plane.close_ws(a).unwrap();
            plane.close().unwrap();
        }
        // Unclosed: a ticket outlives the handle, so the plane ends when the ticket goes.
        drop(plane);
        drop(ticket);
    };
    cycle(true);
    let base = baseline(&fx);
    for closed in [true, false, false, true, false] {
        cycle(closed);
        assert_eq!(settled(&fx, &base), base, "after a plane {}", if closed { "closed" } else { "dropped unclosed" });
    }
}

const DEV: i32 = 0;

struct Stream(CUstream, #[allow(dead_code)] CtxGuard);
impl Stream {
    fn new() -> Stream {
        let (_, ctx) = cuda::primary_context(DEV).unwrap();
        let g = CtxGuard::enter(ctx).unwrap();
        let mut s: CUstream = std::ptr::null_mut();
        let d = cuda::driver().unwrap();
        cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) }).unwrap();
        Stream(s, g)
    }
    fn raw(&self) -> u64 {
        self.0 as u64
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        let d = cuda::driver().unwrap();
        unsafe {
            (d.stream_synchronize)(self.0);
            (d.stream_destroy)(self.0);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum End {
    Closed,
    /// The plane handle goes first; a view of leased bytes holds the last reference.
    Unclosed,
    /// A lease is dropped unreleased: `close` refuses, the plane is collected poisoned.
    Poisoned,
}

fn device_cycle(fx: &Fixture, end: End) {
    let p = Plane::open(PlaneConfig {
        devices: vec![DEV],
        readers: 4,
        staging_buffers: 3,
        staging_bytes: 16 << 20,
        ..Default::default()
    })
    .unwrap();
    let ws = fx.register(&p, None);
    let s = Stream::new();
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, Some(&[0, 1]), 0, false).unwrap().wait().unwrap();
    p.set_vram_budget(DEV, 1 << 30).unwrap();
    p.want(ws, Tier::Device(DEV), Some(&[0]), 0, false).unwrap().wait().unwrap();
    let home = p.acquire(ws, DEV, 0, s.raw(), None).unwrap();
    let view = home.view().unwrap();
    // Region 1 streams from the pinned tier, region 2 from disk through staging.
    let cur = p.stream(ws, DEV, &[1, 2], 1, 1, 0, "census", None).unwrap();
    cur.acquire(1, s.raw()).unwrap().release(s.raw()).unwrap();
    let last = cur.acquire(2, s.raw()).unwrap();
    home.release(s.raw()).unwrap();
    match end {
        End::Closed => {
            last.release(s.raw()).unwrap();
            drop(view);
            cur.close().unwrap();
            p.close().unwrap();
        }
        End::Unclosed => {
            last.release(s.raw()).unwrap();
            drop(p);
            drop(cur);
            drop(view);
        }
        End::Poisoned => {
            drop(last);
            drop(view);
            drop(cur);
            assert!(matches!(p.close(), Err(Error::Poisoned(_))));
            drop(p);
        }
    }
}

#[test]
#[ignore = "GPU"]
fn device_planes_leave_no_descriptors_threads_or_device_memory() {
    let fx = Fixture::new("gpu-census");
    let free = || {
        let (_, ctx) = cuda::primary_context(DEV).unwrap();
        let _g = CtxGuard::enter(ctx).unwrap();
        let (mut free, mut total) = (0usize, 0usize);
        cuda::check("cuMemGetInfo", unsafe { (cuda::driver().unwrap().mem_get_info)(&mut free, &mut total) }).unwrap();
        free
    };
    device_cycle(&fx, End::Closed);
    let (base, vram) = (baseline(&fx), free());
    for end in [End::Closed, End::Unclosed, End::Unclosed, End::Poisoned, End::Closed, End::Poisoned] {
        device_cycle(&fx, end);
        assert_eq!(settled(&fx, &base), base, "after a plane ended {end:?}");
        assert_eq!(free(), vram, "device memory after a plane ended {end:?}");
    }
}
