//! TensorFS v2 checkpoints through `register_checkpoint`: a real v2 store written by its writer,
//! filled into the pinned tier (CPU) and onto a device (GPU, ignored by default), every part
//! read back. The checkpoint has packs, a tensor over two chunks, two components and an empty
//! tensor.
#![cfg(target_os = "linux")]

use std::os::unix::fs::FileExt;

use tensorfs_core::client::regions;
use tensorfs_core::format::Dtype;
use tensorfs_core::store2::{Checkpoint, Store};
use tensorfs_core::writer::{Data, Source};
use varena::cuda::{self, CUstream, CtxGuard};
use varena::{Plane, PlaneConfig, Tier, WsId};

const MIB: usize = 1 << 20;

/// (component, key, bytes), written in this order (the header's).
const TENSORS: &[(&str, &str, usize)] = &[
    ("unet", "blocks.0.a", 9 * MIB + 100),
    ("unet", "blocks.0.b", 8 * MIB + 300),
    ("unet", "blocks.1.w", 70 * MIB + 12),
    ("unet", "blocks.2.a", MIB + 3),
    ("unet", "blocks.2.b", 1000),
    ("unet", "empty", 0),
    ("unet", "norm.weight", 4096),
    ("vae", "conv.weight", 5000),
];

fn bytes_of(key: &str, n: usize) -> Vec<u8> {
    let seed = key.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..n).map(|i| (seed.wrapping_add(i as u32).wrapping_mul(2654435761) >> 13) as u8).collect()
}

struct Fixture {
    root: std::path::PathBuf,
    ck: Checkpoint,
}

impl Fixture {
    fn new(name: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!("varena-ck-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = Store::open(&root).unwrap();
        let mut w = store.writer(None, None).unwrap();
        for &(component, key, n) in TENSORS {
            let b = bytes_of(key, n);
            w.put(component, key, &Dtype::U8, &[n as u64], Data::Plain(Source::Read(&mut &b[..]))).unwrap();
        }
        let ck = store.checkpoint(&w.commit().unwrap()).unwrap();
        Fixture { root, ck }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Every non-empty part of `layout` in `region` (all when None), read by `read(offset, n)`,
/// equals its tensor's bytes.
fn check(p: &Plane, ws: WsId, region: Option<u32>, read: impl Fn(u64, usize) -> Vec<u8>) {
    let layout = p.layout(ws).unwrap();
    let mut seen = 0;
    for part in layout.parts.iter().filter(|x| region.is_none_or(|r| x.region == r)) {
        let (comp_key, _) = part.what.rsplit_once('#').unwrap();
        let (component, key) = comp_key.split_once('/').unwrap();
        let n = TENSORS.iter().find(|t| t.0 == component && t.1 == key).unwrap().2;
        assert_eq!(part.nbytes as usize, n, "{}", part.what);
        if n > 0 {
            assert!(read(part.offset, n) == bytes_of(key, n), "{} differs", part.what);
            seen += 1;
        }
    }
    assert!(seen > 0);
}

#[test]
fn a_v2_checkpoint_fills_the_pinned_tier_exactly() {
    let fx = Fixture::new("pinned");
    let p = Plane::open(PlaneConfig { readers: 2, ..Default::default() }).unwrap();
    let rs = regions(fx.ck.header(), 64 << 20, &[]);
    let ws = p.register_checkpoint("ck", &fx.ck, &rs, None).unwrap();
    assert_eq!(p.layout(ws).unwrap().regions.len(), rs.len());
    p.set_pinned_budget(1 << 30).unwrap();
    p.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    let fd = p.host_fd(ws).unwrap();
    // SAFETY: a borrowed view of the plane's fd for positional reads; never closed here.
    let f = std::mem::ManuallyDrop::new(unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) });
    check(&p, ws, None, |off, n| {
        let mut got = vec![0u8; n];
        f.read_exact_at(&mut got, off).unwrap();
        got
    });
    // One component alone is a weight set of its own tensors.
    let vae: Vec<Vec<String>> = rs.iter().filter(|r| r[0].starts_with("vae/")).cloned().collect();
    let wv = p.register_checkpoint("vae", &fx.ck, &vae, None).unwrap();
    assert!(p.layout(wv).unwrap().parts.iter().all(|x| x.what.starts_with("vae/")));
    p.close().unwrap();
}

#[test]
fn regions_never_span_a_boundary_unless_a_chunk_does() {
    let fx = Fixture::new("boundaries");
    let h = fx.ck.header();
    let unet = |rs: Vec<Vec<String>>| rs.into_iter().filter(|r| r[0].starts_with("unet/")).collect::<Vec<_>>();
    assert_eq!(unet(regions(h, 128 << 20, &[])).len(), 1, "without boundaries the unet fits one region");
    let blocks: Vec<String> = (0..3).map(|i| format!("unet/blocks.{i}")).collect();
    let rs = unet(regions(h, 128 << 20, &blocks));
    let keys = |r: &Vec<String>| r.iter().map(|k| k.trim_start_matches("unet/").to_owned()).collect::<Vec<_>>();
    assert_eq!(rs.iter().map(keys).collect::<Vec<_>>(), vec![
        // An empty tensor has no bytes: it rides in the first region.
        vec!["blocks.0.a", "blocks.0.b", "empty"],
        vec!["blocks.1.w"],
        // blocks.2 and norm share a pack: a chunk keeps them together.
        vec!["blocks.2.a", "blocks.2.b", "norm.weight"],
    ]);
}

#[test]
#[ignore = "GPU"]
fn a_v2_checkpoint_reaches_the_device_exactly() {
    let fx = Fixture::new("device");
    let (_, ctx) = cuda::primary_context(0).unwrap();
    let _g = CtxGuard::enter(ctx).unwrap();
    let d = cuda::driver().unwrap();
    let mut s: CUstream = std::ptr::null_mut();
    cuda::check("cuStreamCreate", unsafe { (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING) }).unwrap();
    let p = Plane::open(PlaneConfig { devices: vec![0], ..Default::default() }).unwrap();
    let ws = p.register_checkpoint("ck", &fx.ck, &regions(fx.ck.header(), 64 << 20, &[]), None).unwrap();
    p.set_vram_budget(0, 1 << 30).unwrap();
    p.want(ws, Tier::Device(0), None, 0, false).unwrap().wait().unwrap();
    let layout = p.layout(ws).unwrap();
    for r in 0..layout.regions.len() as u32 {
        let l = p.acquire(ws, 0, r, s as u64, None).unwrap();
        let mut got = vec![0u8; l.nbytes as usize];
        cuda::check("cuMemcpyDtoHAsync", unsafe { (d.memcpy_dtoh_async)(got.as_mut_ptr() as *mut libc::c_void, l.ptr, got.len(), s) }).unwrap();
        cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(s) }).unwrap();
        let base = layout.regions[r as usize].offset;
        check(&p, ws, Some(r), |off, n| got[(off - base) as usize..(off - base) as usize + n].to_vec());
        l.release(s as u64).unwrap();
    }
    let st = p.stats();
    println!("device: {:?}", st.devices[0].counters);
    assert!(st.poisoned.is_none());
    cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(s) }).unwrap();
    p.close().unwrap();
}
