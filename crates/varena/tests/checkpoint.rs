//! TensorFS v2 checkpoints through `register`: a real v2 store written by its writer, bound on
//! the host core from its chunk files and from a sealed host layout, every part read back. The
//! checkpoint has packs, a tensor over two chunks, two components and an empty tensor. The GPU
//! core's proof runs on a rental (`scripts/varena-gpu.py`).
#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::sync::Arc;

use tensorfs_core::client::regions;
use tensorfs_core::format::Dtype;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store2::{Checkpoint, Store};
use tensorfs_core::writer::{Data, Source};
use varena::checkpoint::Ck;
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

/// Every chunk of `ck`, opened: what a grant hands over.
fn granted(ck: &Checkpoint) -> Vec<(ObjectRef, std::fs::File)> {
    let chunks = ck.header().chunks.iter().enumerate();
    chunks.map(|(i, c)| (ObjectRef { sha256: c.sha256.clone(), length: c.length }, std::fs::File::open(ck.chunk_path(i)).unwrap())).collect()
}

#[test]
fn a_v2_checkpoint_binds_on_the_host_exactly_from_its_chunk_files() {
    let fx = Fixture::new("host");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let ws = Host.register("ck", &fx.ck, &rs, None, None).unwrap();
    assert_eq!(ws.weights.layout.regions.len(), rs.len());
    check(&ws);
    assert!(ws.bind(0).unwrap().writable);
    // One component alone is a weight set of its own tensors.
    let vae: Vec<Vec<String>> = rs.iter().filter(|r| r[0].starts_with("vae/")).cloned().collect();
    assert!(Host.register("vae", &fx.ck, &vae, None, None).unwrap().weights.parts.iter().all(|p| p.component == "vae"));
    // A window is a weight set of its regions' parts alone, read exactly.
    let window = regions(fx.ck.header(), 64 << 20, &blocks(), &["unet/blocks.2".into()]);
    let ww = Host.register("window", &fx.ck, &window, None, None).unwrap();
    let parts: Vec<&str> = ww.weights.parts.iter().map(|p| p.key.as_str()).collect();
    assert_eq!(parts, ["blocks.2.a", "blocks.2.b", "norm.weight"]);
    check_parts(&ww, 3);
    assert!(Host.register("bad", &fx.ck, &[vec!["unet/blocks.9".into()]], None, None).is_err());
}

#[test]
fn a_v2_checkpoint_binds_from_the_chunk_files_a_grant_hands_over_and_no_others() {
    let fx = Fixture::new("granted");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    check(&Host.register("granted", &fx.ck, &rs, None, Some(granted(&fx.ck))).unwrap());
    let none = Host.register("none", &fx.ck, &rs, None, Some(Vec::new())).unwrap();
    assert!((0..rs.len() as u32).any(|r| none.bind(r).is_err()), "an object no grant handed over is never read");
}

#[test]
fn a_header_parsed_alone_binds_and_seals_from_its_grant_only() {
    let fx = Fixture::new("parsed");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let parsed = Ck { header: fx.ck.header(), stored: None };
    check(&Host.register("parsed", parsed, &rs, None, Some(granted(&fx.ck))).unwrap());
    assert!(matches!(Host.register("bare", parsed, &rs, None, None), Err(Error::Invalid(_))), "no store, no grant: nothing to read");
    let fd = memory::seal(parsed, &rs, "parsed", 2, Some(granted(&fx.ck))).unwrap();
    check(&Host.register("sealed", parsed, &rs, Some(fd.as_raw_fd()), None).unwrap());
}

#[cfg(target_os = "linux")] // sealed host layouts are Linux-only
#[test]
fn a_v2_checkpoint_binds_on_the_host_exactly_from_a_sealed_layout() {
    let fx = Fixture::new("sealed");
    let rs = regions(fx.ck.header(), 64 << 20, &[], &[]);
    let fd = memory::seal(&fx.ck, &rs, "ck", 2, None).unwrap();
    let ws = Host.register("ck", &fx.ck, &rs, Some(fd.as_raw_fd()), None).unwrap();
    check(&ws);
    assert!(!ws.bind(0).unwrap().writable, "views of the machine's layout are read-only");
    let mut moved = rs.clone();
    let tail = moved[0].pop().unwrap();
    moved[1].push(tail);
    assert!(Host.register("ck", &fx.ck, &moved, Some(fd.as_raw_fd()), None).is_err(), "another grouping is refused");
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
    check(&Host.register("runs", &fx.ck, &rs, None, None).unwrap());
    let fd = memory::seal(&fx.ck, &rs, "runs", 2, None).unwrap();
    check(&Host.register("runs", &fx.ck, &rs, Some(fd.as_raw_fd()), None).unwrap());
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
