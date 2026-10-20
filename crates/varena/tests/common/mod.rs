//! A real TensorFS store in a temp dir holding a small odd-sized model, written by its writer
//! and opened as a checkpoint. Shared by the host-tier tests.
#![allow(dead_code)]

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tensorfs::format::{Dtype, ObjectRef};
use tensorfs::io::Span;
use tensorfs::store::Store;
use tensorfs::writer::{Data, Source};
use tensorfs::{Checkpoint, Client, Config, Open};
use varena::host::HostMem;
use varena::layout::Layout;

pub const MIB: usize = 1 << 20;

/// (key, bytes): odd sizes on purpose.
pub const TENSORS: &[(&str, usize)] = &[
    ("a.weight", 3 * MIB + 100),
    ("a.bias", 200),
    ("b.weight", 70 * MIB + 12),
    ("c.weight", MIB + 3),
    ("c.norm", 1000),
];

pub fn regions() -> Vec<Vec<String>> {
    [&["a.weight", "a.bias"][..], &["b.weight"], &["c.weight", "c.norm"]]
        .iter()
        .map(|r| r.iter().map(|k| format!("unet/{k}")).collect())
        .collect()
}

pub fn bytes_of(key: &str, n: usize) -> Vec<u8> {
    let seed = key.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..n).map(|i| (seed.wrapping_add(i as u32).wrapping_mul(2654435761) >> 13) as u8).collect()
}

pub fn scratch(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tfs-plane-{name}-{}-{n}", std::process::id()))
}

/// `tensors` (component, key, bytes) as U8 tensors in this order, written into the store at
/// `root` and opened from it.
pub fn checkpoint(root: &Path, tensors: &[(&str, &str, Vec<u8>)]) -> (Store, Checkpoint) {
    let store = Store::open(root).unwrap();
    let mut w = store.writer(None, None).unwrap();
    for (component, key, bytes) in tensors {
        let data = Data::Plain(Source::Read(&mut &bytes[..]));
        w.get(component).put(key, &Dtype::U8, &[bytes.len() as u64], data).unwrap();
    }
    let header = w.commit().unwrap();
    let client = Client::new(Config { cache_dir: root.to_path_buf(), ..Config::from_env() }).unwrap();
    let open = Open { local_files_only: true, ..Default::default() };
    let reference = format!("sha256:{}", header.sha256);
    let ck = tensorfs::raw::runtime().unwrap().block_on(client.open(&reference, &open)).unwrap();
    (store, ck)
}

pub struct Fixture {
    pub root: PathBuf,
    pub store: Store,
    pub ck: Checkpoint,
    pub objects: Vec<ObjectRef>,
}

impl Fixture {
    pub fn new(name: &str) -> Fixture {
        let root = scratch(name);
        let tensors: Vec<_> = TENSORS.iter().map(|&(k, n)| ("unet", k, bytes_of(k, n))).collect();
        let (store, ck) = checkpoint(&root, &tensors);
        let objects = ck.chunks().to_vec();
        Fixture { root, store, ck, objects }
    }

    pub fn plan(&self) -> Vec<Span> {
        self.ck.spans(&regions()).unwrap()
    }

    pub fn source(&self) -> varena::io::Source {
        self.ck.source(true)
    }

    pub fn path(&self, object: &ObjectRef) -> PathBuf {
        self.store.path(object).unwrap()
    }

    /// Evict our own blob files from the page cache so a fill has to go to disk. Written back
    /// first: a dirty page stays whatever the advice.
    pub fn drop_cache(&self) {
        for o in &self.objects {
            let f = std::fs::File::open(self.path(o)).unwrap();
            f.sync_all().unwrap();
            // SAFETY: advice on our own descriptor.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Page-cache pages resident across our blob files (mincore; never faults them in).
    pub fn cached_pages(&self) -> usize {
        let mut n = 0;
        for o in &self.objects {
            let f = std::fs::File::open(self.path(o)).unwrap();
            let len = o.length as usize;
            // SAFETY: read-only shared mapping of our file, unmapped below.
            let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
            assert_ne!(p, libc::MAP_FAILED);
            let mut v = vec![0u8; len.div_ceil(4096)];
            // SAFETY: vector sized for the page count.
            assert_eq!(unsafe { libc::mincore(p, len, v.as_mut_ptr()) }, 0);
            unsafe { libc::munmap(p, len) };
            n += v.iter().filter(|b| **b & 1 == 1).count();
        }
        n
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Every part's bytes in `host` equal the source tensor's bytes.
pub fn check_host(host: &HostMem, layout: &Layout, region: Option<u32>) {
    for p in layout.parts.iter().filter(|p| region.is_none_or(|r| p.region == r)) {
        let key = p.what.trim_start_matches("unet/").trim_end_matches("#value");
        let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
        // SAFETY: the part lies inside the live mapping.
        let got = unsafe { std::slice::from_raw_parts(host.base().add(p.offset as usize), n) };
        assert!(got == bytes_of(key, n), "{} differs in the host layout", p.what);
    }
}
