//! A real TensorFS store in a temp dir holding a small odd-sized model, a real read lease and
//! plan. Shared by the host-tier and GPU integration tests.
#![allow(dead_code)]

use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Body, Header, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::meta::Meta;
use tensorfs_core::read::{self, plan_for_traversal, ReadPlan};
use tensorfs_core::registry;
use tensorfs_core::store::{Fault, Store};
use tensorfs_plane::{Plane, PlaneConfig, WsId};

pub const MIB: usize = 1 << 20;

/// (key, bytes): odd sizes on purpose; one tensor spans two 64 MiB objects, one is inline.
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

pub struct Fixture {
    pub root: PathBuf,
    pub store: Arc<Store>,
    pub meta: Arc<Meta>,
    pub header: Header,
    pub objects: Vec<ObjectRef>,
}

impl Fixture {
    pub fn new(name: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "tfs-plane-{name}-{}-{}",
            std::process::id(),
            tensorfs_core::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let plain = registry::seeds().into_iter().find(|s| s.alias == "plain/1").unwrap().spec;
        let mut objects = Vec::new();
        let tensors = TENSORS
            .iter()
            .map(|&(key, n)| {
                let b = bytes_of(key, n);
                let part = Part::plan(Dtype::U8, vec![n as u64], &b);
                if let Body::Segments(segs) = &part.body {
                    let mut at = 0usize;
                    for s in segs {
                        let chunk = &b[at..at + s.length as usize];
                        store.put_stream(&mut &chunk[..], Some(s), &Fault::default()).unwrap();
                        objects.push(s.clone());
                        at += s.length as usize;
                    }
                }
                (
                    key.to_string(),
                    Tensor {
                        dtype: Dtype::U8,
                        shape: vec![n as u64],
                        encoding: plain.object_id(),
                        parts: vec![("value".into(), part)],
                    },
                )
            })
            .collect();
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![plain],
            components: vec![("unet".into(), tensors)],
        };
        let header = Header::parse(&header.canonical_bytes().unwrap()).unwrap();
        let meta = Arc::new(Meta::open(&store).unwrap());
        Fixture {
            root,
            store: Arc::new(store),
            meta,
            header,
            objects,
        }
    }

    pub fn plan(&self) -> ReadPlan {
        let traversal: Vec<(String, String)> =
            TENSORS.iter().map(|(k, _)| ("unet".into(), k.to_string())).collect();
        plan_for_traversal(&self.header, &traversal, &["unet".into()], 16 << 20).unwrap()
    }

    pub fn register(&self, plane: &Plane, host_fd: Option<i32>) -> WsId {
        let (lease, _) = read::acquire(&self.store, &self.meta, "test", self.objects.clone()).unwrap();
        let src = plane.source(self.store.clone(), self.meta.clone(), lease);
        plane.register("unet", src, &self.plan(), &regions(), host_fd).unwrap()
    }

    /// Live TensorFS holds (lease files) in the store.
    pub fn holds(&self) -> usize {
        std::fs::read_dir(self.root.join("tmp/leases"))
            .map(|d| d.filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|x| x == "lease")).count())
            .unwrap_or(0)
    }

    /// Evict our own blob files from the page cache so a fill has to go to disk. Written back
    /// first: a dirty page stays whatever the advice.
    pub fn drop_cache(&self) {
        for o in &self.objects {
            let f = std::fs::File::open(self.store.blob_path(&o.sha256)).unwrap();
            f.sync_all().unwrap();
            // SAFETY: advice on our own descriptor.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Page-cache pages resident across our blob files (mincore; never faults them in).
    pub fn cached_pages(&self) -> usize {
        let mut n = 0;
        for o in &self.objects {
            let f = std::fs::File::open(self.store.blob_path(&o.sha256)).unwrap();
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

pub fn host_only() -> Plane {
    Plane::open(PlaneConfig {
        readers: 2,
        ..Default::default()
    })
    .unwrap()
}

/// Every part's bytes in the memfd equal the source tensor's bytes.
pub fn check_bytes(plane: &Plane, ws: WsId, region: Option<u32>) {
    let layout = plane.layout(ws).unwrap();
    let fd = plane.host_fd(ws).unwrap();
    // SAFETY: a borrowed view of the plane's fd for positional reads; never closed here.
    let f = std::mem::ManuallyDrop::new(unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) });
    for p in layout.parts.iter().filter(|p| region.is_none_or(|r| p.region == r)) {
        let key = p.what.trim_start_matches("unet/").trim_end_matches("#value");
        let n = TENSORS.iter().find(|(k, _)| *k == key).unwrap().1;
        let mut got = vec![0u8; n];
        f.read_exact_at(&mut got, p.offset).unwrap();
        assert!(got == bytes_of(key, n), "{} differs in the pinned tier", p.what);
    }
}

