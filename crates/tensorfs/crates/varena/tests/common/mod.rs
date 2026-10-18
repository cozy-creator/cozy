//! A real TensorFS v2 store in a temp dir holding a small odd-sized checkpoint: two components,
//! a tensor split over two chunks, packs, a chunk two blocks share, and an empty tensor.
#![allow(dead_code)]

use std::os::fd::AsRawFd;
use std::path::PathBuf;

use tensorfs_core::client;
use tensorfs_core::format::Dtype;
use tensorfs_core::store2::{Checkpoint, Store};
use tensorfs_core::writer::Data;

pub const MIB: usize = 1 << 20;

/// (component, key, bytes), in the writer's natural order. Chunks: 0 = blocks.0 (17 MiB, closes
/// at the next block); 1, 2 = blocks.1.w (64 MiB + 6 MiB); 3 = blocks.2, blocks.3 and norm
/// (one pack under 16 MiB); 4 = vae.
pub const TENSORS: &[(&str, &str, usize)] = &[
    ("unet", "blocks.0.a", 9 * MIB + 100),
    ("unet", "blocks.0.b", 8 * MIB + 300),
    ("unet", "blocks.1.w", 70 * MIB + 12),
    ("unet", "blocks.2.a", MIB + 3),
    ("unet", "blocks.2.b", 1000),
    ("unet", "blocks.3.a", 2 * MIB + 7),
    ("unet", "empty", 0),
    ("unet", "norm.weight", 4096),
    ("vae", "conv.weight", 5000),
];

pub fn bytes_of(key: &str, n: usize) -> Vec<u8> {
    let seed = key.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..n).map(|i| (seed.wrapping_add(i as u32).wrapping_mul(2654435761) >> 13) as u8).collect()
}

pub fn size_of(component: &str, key: &str) -> usize {
    TENSORS.iter().find(|t| t.0 == component && t.1 == key).unwrap().2
}

pub struct Fixture {
    pub root: PathBuf,
    pub ck: Checkpoint,
}

impl Fixture {
    pub fn new(name: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!("varena-{name}-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = Store::open(&root).unwrap();
        let mut w = store.writer(None, None).unwrap();
        for &(component, key, n) in TENSORS {
            let b = bytes_of(key, n);
            w.put(component, key, &Dtype::U8, &[n as u64], Data::Plain(&mut &b[..])).unwrap();
        }
        let m = w.commit().unwrap();
        let ck = store.checkpoint(&m).unwrap();
        assert_eq!(ck.header().chunks.len(), 5, "the fixture's chunk plan changed");
        Fixture { root, ck }
    }

    /// `regions(ck, 64 MiB)`: [[0], [1, 2], [3], [4]].
    pub fn regions(&self) -> Vec<Vec<u32>> {
        client::regions(self.ck.header(), 64 << 20, &[])
    }

    /// Evict the chunk files from the page cache so a fill goes to disk. Written back first: a
    /// dirty page stays whatever the advice.
    pub fn drop_cache(&self) {
        for p in self.ck.chunk_paths() {
            let f = std::fs::File::open(p).unwrap();
            f.sync_all().unwrap();
            // SAFETY: advice on our own descriptor.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Every part of region `r`, read from `got` (the region's bytes from its first), equals its
/// tensor's bytes.
pub fn check_region(layout: &varena::Layout, r: u32, got: &[u8]) {
    let off = layout.regions[r as usize].offset;
    let mut n = 0;
    for p in layout.parts.iter().filter(|p| p.region == r) {
        let at = (p.offset - off) as usize;
        let want = bytes_of(&p.key, size_of(&p.component, &p.key));
        assert!(got[at..at + want.len()] == want[..], "{} differs", p.what());
        n += 1;
    }
    assert!(n > 0, "region {r} holds no part");
}

/// A sealed host layout of `layout`, filled from the chunk files as an owner would: (memfd, fill).
/// `held`: the regions it holds (None: all).
pub fn sealed(fx: &Fixture, layout: &varena::Layout, held: Option<&[u32]>) -> (std::os::fd::OwnedFd, varena::host::SealedFill) {
    let (fd, fill) = varena::host::open_sealed("fixture", layout, held).unwrap();
    for (r, region) in layout.regions.iter().enumerate() {
        if fill.hole(r as u32) {
            continue;
        }
        for c in &region.chunks {
            let bytes = std::fs::read(&fx.ck.chunk_paths()[c.index as usize]).unwrap();
            // SAFETY: inside the writable mapping; the owner's fill is this thread's alone.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), fill.base().add(c.offset as usize), bytes.len()) };
        }
        fill.set_ready(r as u32);
    }
    (fd, fill)
}
