//! D3's measurement on a real checkpoint: an executor restart re-uploading a model from a sealed
//! host layout (an owner's memfd) to the device, against a cold read of the chunk files.
//! `VARENA_BENCH_STORE` and `VARENA_BENCH_MANIFEST` name a v2 store and checkpoint (locations);
//! the regions that fit the device less 1 GiB are used. Prints one JSON line per cell.
//! `cargo test -p varena --test reupload -- --ignored --nocapture`
#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::time::Instant;

use tensorfs_core::client::regions;
use tensorfs_core::store2::Store;
use varena::cuda::{self, CtxGuard};
use varena::{Host, Layout, Plane, PlaneConfig};

const DEV: i32 = 0;

fn cell(name: &str, bytes: u64, secs: f64, p: &Plane, extra: &str) {
    let st = p.stats();
    let d = &st.devices[0].counters;
    let gbps = |b: u64, ns: u64| b as f64 / ns.max(1) as f64;
    println!(
        "{{\"cell\":\"{name}\",\"bytes\":{bytes},\"s\":{secs:.3},\"gb_s\":{:.2},\"dma_gb_s\":{:.2},\"bounce_gb_s\":{:.2},\"disk_gb_s\":{:.2},\"register_failed\":{},\"direct_bytes\":{},\"cached_bytes\":{}{extra}}}",
        bytes as f64 / secs / 1e9,
        gbps(d.host_copy_bytes, d.host_copy_ns),
        gbps(d.bounce_copy_bytes, d.bounce_copy_ns),
        gbps(d.disk_copy_bytes, d.disk_copy_ns),
        st.io.register_failed,
        st.io.direct_bytes,
        st.io.cached_bytes,
    );
}

#[test]
#[ignore = "GPU and a real checkpoint"]
fn restart_reupload_from_a_sealed_layout() {
    let store = Store::open(std::env::var("VARENA_BENCH_STORE").expect("VARENA_BENCH_STORE")).unwrap();
    let reference = std::env::var("VARENA_BENCH_MANIFEST").expect("VARENA_BENCH_MANIFEST");
    let ck = store.checkpoint(&store.resolve(&reference, None).unwrap().manifest).unwrap();
    let t = Instant::now();
    let (_, ctx) = cuda::primary_context(DEV).unwrap();
    let (free, _) = {
        let _g = CtxGuard::enter(ctx).unwrap();
        varena::device::mem_info().unwrap()
    };
    println!("{{\"cell\":\"context\",\"s\":{:.3}}}", t.elapsed().as_secs_f64());
    // The regions that fit: execution order, until free memory less 1 GiB.
    let all = regions(ck.header(), 64 << 20, &[]);
    let full = Layout::build(ck.header(), &all).unwrap();
    let mut used = 0;
    let n = full.regions.iter().take_while(|r| {
        used += r.span;
        used + (1 << 30) <= free
    });
    let picked: Vec<Vec<u32>> = all[..n.count()].to_vec();
    let layout = Layout::build(ck.header(), &picked).unwrap();
    let bytes: u64 = layout.regions.iter().map(|r| r.nbytes).sum();
    let cfg = || PlaneConfig { devices: vec![DEV], ..Default::default() };

    // Cold chunk files: what a first load reads.
    for p in ck.chunk_paths() {
        let f = std::fs::File::open(p).unwrap();
        unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
    let p = Plane::open(cfg()).unwrap();
    let t = Instant::now();
    let ws = p.register(&ck, &picked, Host::PageCache { direct: true }).unwrap();
    p.want(ws, DEV, None).unwrap().wait().unwrap();
    cell("files-cold", bytes, t.elapsed().as_secs_f64(), &p, "");
    p.close().unwrap();

    // The owner's sealed layout, filled once (TensorD's host cache in Cozy).
    let t = Instant::now();
    let (fd, fill) = varena::host::open_sealed("reupload", &layout, None).unwrap();
    for (r, region) in layout.regions.iter().enumerate() {
        for c in &region.chunks {
            let bytes = std::fs::read(&ck.chunk_paths()[c.index as usize]).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), fill.base().add(c.offset as usize), bytes.len()) };
        }
        fill.set_ready(r as u32);
    }
    println!("{{\"cell\":\"owner-fill\",\"bytes\":{bytes},\"s\":{:.3}}}", t.elapsed().as_secs_f64());

    // The mechanism alone: page-lock the layout's RAM and copy it in one DMA (what the plane
    // does where the driver locks read-only memory; GeForce cards refuse and bounce instead).
    {
        let _g = CtxGuard::enter(ctx).unwrap();
        let d = cuda::driver().unwrap();
        let n = layout.nbytes as usize;
        let t = Instant::now();
        cuda::check("cuMemHostRegister", unsafe { (d.mem_host_register)(fill.base() as *mut libc::c_void, n, cuda::MEMHOSTREGISTER_PORTABLE) }).unwrap();
        let pinned = t.elapsed().as_secs_f64();
        let va = varena::device::reserve_va(layout.nbytes).unwrap();
        let (dev, _) = cuda::primary_context(DEV).unwrap();
        let mem = varena::device::create(&cuda::AllocationProp::device(dev, false), layout.nbytes).unwrap();
        varena::device::map_span(va, &[mem], dev, cuda::MEM_ACCESS_PROT_READWRITE).unwrap();
        let t = Instant::now();
        cuda::check("cuMemcpyHtoD", unsafe { (d.memcpy_htod_async)(va, fill.base() as *const libc::c_void, n, std::ptr::null_mut()) }).unwrap();
        cuda::check("cuCtxSynchronize", unsafe { (d.ctx_synchronize)() }).unwrap();
        let dma = t.elapsed().as_secs_f64();
        println!(
            "{{\"cell\":\"pin-dma\",\"bytes\":{bytes},\"pin_s\":{pinned:.3},\"pin_gb_s\":{:.2},\"dma_s\":{dma:.3},\"dma_gb_s\":{:.2}}}",
            n as f64 / pinned / 1e9,
            n as f64 / dma / 1e9
        );
        varena::device::unmap_span(va, layout.nbytes).unwrap();
        varena::device::release(mem.handle).unwrap();
        varena::device::free_va(va, layout.nbytes).unwrap();
        cuda::check("cuMemHostUnregister", unsafe { (d.mem_host_unregister)(fill.base() as *mut libc::c_void) }).unwrap();
    }

    // The restart: a fresh plane opens, registers the layout, page-locks and copies it.
    for round in 0..2 {
        let t = Instant::now();
        let p = Plane::open(cfg()).unwrap();
        let opened = t.elapsed().as_secs_f64();
        let ws = p.register(&ck, &picked, Host::Sealed(fd.as_raw_fd())).unwrap();
        p.want(ws, DEV, None).unwrap().wait().unwrap();
        cell(&format!("restart-{round}"), bytes, t.elapsed().as_secs_f64(), &p, &format!(",\"open_s\":{opened:.3}"));
        // The same plane again: the layout stays page-locked, so this is the copy alone.
        p.drop_regions(ws, DEV, None).unwrap();
        let t = Instant::now();
        p.want(ws, DEV, None).unwrap().wait().unwrap();
        cell(&format!("refill-{round}"), bytes, t.elapsed().as_secs_f64(), &p, "");
        p.close().unwrap();
    }
    drop(fill);
}
