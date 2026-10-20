//! TensorFS checkpoints through `register`: a real store written by its writer, bound on the
//! host core from its chunks and from a sealed host layout, every part read back. The
//! checkpoint has packs, a tensor over two chunks, two components and an empty tensor. The GPU
//! core's proof runs on a rental (`scripts/varena-gpu.py`).
#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::sync::Arc;

mod common;

use tensorfs::raw::regions;
use tensorfs::Checkpoint;
use varena::{cuda, memory, Error, Gpu, Host, HostSet};

const MIB: usize = 1 << 20;

fn blocks() -> Vec<String> {
    (0..3).map(|i| format!("unet/blocks.{i}")).collect()
}

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
        let root = common::scratch(&format!("ck-{name}"));
        let tensors: Vec<_> = TENSORS.iter().map(|&(c, k, n)| (c, k, bytes_of(k, n))).collect();
        let (_, ck) = common::checkpoint(&root, &tensors);
        Fixture { root, ck }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn check(ws: &Arc<HostSet>) {
    check_parts(ws, TENSORS.iter().filter(|t| t.2 > 0).count());
}

/// Every part, bound region by region, equals its tensor's bytes and carries its dtype and
/// shape; the first bind of a region is fresh, the next is not; `want` parts have bytes.
fn check_parts(ws: &Arc<HostSet>, want: usize) {
    let mut seen = 0;
    for r in 0..ws.weights.layout.regions.len() as u32 {
        let pin = ws.bind(r).unwrap();
        assert!(pin.fresh && !ws.bind(r).unwrap().fresh);
        for part in ws.weights.parts.iter().filter(|p| p.region == r) {
            let n = TENSORS.iter().find(|t| t.0 == part.component && t.1 == part.key).unwrap().2;
            assert_eq!((part.nbytes as usize, part.dtype.as_str(), part.shape.as_slice(), part.role.as_str()), (n, "U8", &[n as u64][..], ""));
            // SAFETY: the part lies inside the region the pin keeps.
            let got = unsafe { std::slice::from_raw_parts((pin.ptr + part.offset) as *const u8, n) };
            assert!(got == bytes_of(&part.key, n), "{}/{} differs", part.component, part.key);
            seen += (n > 0) as usize;
        }
    }
    assert_eq!(seen, want);
}

#[test]
fn a_checkpoint_binds_on_the_host_exactly_from_its_chunks() {
    let fx = Fixture::new("host");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let ws = Host.register("ck", &fx.ck, &rs, None).unwrap();
    assert_eq!(ws.weights.layout.regions.len(), rs.len());
    check(&ws);
    assert!(ws.bind(0).unwrap().writable);
    // One component alone is a weight set of its own tensors.
    let vae: Vec<Vec<String>> = rs.iter().filter(|r| r[0].starts_with("vae/")).cloned().collect();
    assert!(Host.register("vae", &fx.ck, &vae, None).unwrap().weights.parts.iter().all(|p| p.component == "vae"));
    // A window is a weight set of its regions' parts alone, read exactly.
    let window = regions(fx.ck.header(), 64 << 20, &blocks(), &["unet/blocks.2".into()]);
    let ww = Host.register("window", &fx.ck, &window, None).unwrap();
    let parts: Vec<&str> = ww.weights.parts.iter().map(|p| p.key.as_str()).collect();
    assert_eq!(parts, ["blocks.2.a", "blocks.2.b", "norm.weight"]);
    check_parts(&ww, 3);
    assert!(Host.register("bad", &fx.ck, &[vec!["unet/blocks.9".into()]], None).is_err());
}

#[cfg(target_os = "linux")] // sealed host layouts are Linux-only
#[test]
fn a_checkpoint_binds_on_the_host_exactly_from_a_sealed_layout() {
    let fx = Fixture::new("sealed");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let fd = memory::seal(&fx.ck, &rs, "ck", 2).unwrap();
    let carried = varena::host::sealed_regions(fd.as_raw_fd()).unwrap();
    assert_eq!(carried, rs, "a sealed layout carries the regions it was built from");
    let ws = Host.register("ck", &fx.ck, &carried, Some(fd.as_raw_fd())).unwrap();
    check(&ws);
    assert!(!ws.bind(0).unwrap().writable, "views of the machine's layout are read-only");
    let mut moved = rs.clone();
    let tail = moved[0].pop().unwrap();
    moved[1].push(tail);
    assert!(Host.register("ck", &fx.ck, &moved, Some(fd.as_raw_fd())).is_err(), "another grouping is refused");
}

/// The glue's regions: header-order runs cut where the owning block changes. blocks.2.b and norm
/// share a pack, so that chunk is read once per region, from the files and into a sealed layout.
#[cfg(target_os = "linux")]
#[test]
fn a_chunk_two_regions_share_is_read_per_region() {
    let fx = Fixture::new("runs");
    let chunk = |key: &str| fx.ck.header().tensor("unet", key).unwrap().locations()[0].1.chunk;
    assert_eq!(chunk("blocks.2.b"), chunk("norm.weight"));
    let mut rs: Vec<Vec<String>> = Vec::new();
    let mut last = None;
    for &(c, k, _) in TENSORS {
        let owner = (c, blocks().into_iter().find(|b| format!("{c}/{k}").starts_with(&format!("{b}."))));
        if last.as_ref() != Some(&owner) {
            rs.push(Vec::new());
            last = Some(owner);
        }
        rs.last_mut().unwrap().push(format!("{c}/{k}"));
    }
    assert!(rs.contains(&vec!["unet/blocks.2.a".into(), "unet/blocks.2.b".into()]), "{rs:?}");
    check(&Host.register("runs", &fx.ck, &rs, None).unwrap());
    let fd = memory::seal(&fx.ck, &rs, "runs", 2).unwrap();
    check(&Host.register("runs", &fx.ck, &rs, Some(fd.as_raw_fd())).unwrap());
}

#[test]
fn a_gpu_takes_whole_pages_and_says_why_it_cannot_open() {
    assert!(matches!(Gpu::open(0, 3 << 20), Err(Error::Invalid(_))), "chunks are whole 2 MiB pages");
    if !cuda::available() {
        assert!(matches!(Gpu::open(0, 2 << 20), Err(Error::CudaUnavailable(_))));
    }
}

#[test]
fn regions_never_span_a_boundary_unless_a_chunk_does() {
    let fx = Fixture::new("boundaries");
    let h = fx.ck.header();
    let unet = |rs: Vec<Vec<String>>| rs.into_iter().filter(|r| r[0].starts_with("unet/")).collect::<Vec<_>>();
    assert_eq!(unet(regions(h, 128 << 20, &[], &[])).len(), 1, "without boundaries the unet fits one region");
    let rs = unet(regions(h, 128 << 20, &blocks(), &[]));
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
fn a_window_gets_the_regions_the_full_call_gives_its_tensors() {
    let fx = Fixture::new("window");
    let h = fx.ck.header();
    for max in [MIB as u64, 64 << 20, 128 << 20] {
        let full = regions(h, max, &blocks(), &[]);
        for only in ["unet/blocks.0", "unet/blocks.1", "unet/blocks.2", "vae/conv.weight"] {
            // The full call's regions holding a tensor of `only` (the empty tensor is no window's).
            let want: Vec<Vec<String>> = full
                .iter()
                .filter(|r| r.iter().any(|t| t == only || t.starts_with(&format!("{only}."))))
                .map(|r| r.iter().filter(|t| *t != "unet/empty").cloned().collect())
                .collect();
            assert_eq!(regions(h, max, &blocks(), &[only.into()]), want, "{only} at {max}");
        }
    }
    // blocks.2 shares a pack with norm: a window of it reads the whole chunk, so holds norm too.
    assert!(regions(h, 64 << 20, &blocks(), &["unet/blocks.2".into()]).concat().contains(&"unet/norm.weight".to_owned()));
    // Two windows: each block's regions, never one across them.
    let two = regions(h, 128 << 20, &blocks(), &["unet/blocks.0".into(), "unet/blocks.1".into()]);
    assert_eq!(two, [vec!["unet/blocks.0.a", "unet/blocks.0.b"], vec!["unet/blocks.1.w"]]);
    assert!(regions(h, 64 << 20, &blocks(), &["unet/nothing".into()]).is_empty());
}

#[test]
fn a_seal_logs_its_disk_copy() {
    // The copy log is process-global; isolate this operation from parallel tests.
    if std::env::var_os("VARENA_COPY_LOG_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "a_seal_logs_its_disk_copy", "--nocapture"])
            .env("VARENA_COPY_LOG_CHILD", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    let fx = Fixture::new("logged");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    varena::ceiling::take();
    memory::seal(&fx.ck, &rs, "logged", 2).unwrap();
    let mut expected = rs.iter().map(|names| names.iter().map(|name| {
        TENSORS.iter().find(|(c, k, _)| format!("{c}/{k}") == *name).unwrap().2 as u64
    }).sum::<u64>()).filter(|bytes| *bytes > 0).collect::<Vec<_>>();
    let copies = varena::ceiling::take();
    assert!(copies.iter().all(|c| c.kind == "disk" && c.ns > 0), "{copies:?}");
    let mut observed = copies.iter().map(|c| c.bytes).collect::<Vec<_>>();
    expected.sort_unstable(); observed.sort_unstable();
    assert_eq!(observed, expected, "each region's exact payload bytes are measured once");
}

/// On a GPU only: `give` takes a slot of a set in use that no bound block holds, never a bound
/// one, and the next stream maps a slot again.
#[test]
fn give_takes_a_busy_sets_unbound_slot_never_a_bound_one() {
    if !cuda::available() {
        return;
    }
    let fx = Fixture::new("give");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let gpu = Gpu::open(0, 2 << 20).unwrap();
    let ws = gpu.register("give", &fx.ck, &rs, None).unwrap();
    ws.home(&[]).unwrap();
    ws.busy(true);
    let slots = || gpu.stats().unwrap().slots;
    let sync = || {
        let (_, ctx) = cuda::primary_context(0).unwrap();
        let _c = cuda::CtxGuard::enter(ctx).unwrap();
        // SAFETY: the legacy stream of our current context.
        cuda::check("cuStreamSynchronize", unsafe { (cuda::driver().unwrap().stream_synchronize)(std::ptr::null_mut()) }).unwrap();
    };
    let pin = ws.bind(0, 0).unwrap();
    sync();
    gpu.give(u64::MAX).unwrap();
    assert_eq!(slots(), 1, "a bound slot is never given");
    drop(pin);
    sync();
    gpu.give(u64::MAX).unwrap();
    assert_eq!(slots(), 0, "an unbound slot of a busy set is given");
    let _pin = ws.bind(0, 0).unwrap();
    assert_eq!(slots(), 1, "the next stream maps one again");
}

