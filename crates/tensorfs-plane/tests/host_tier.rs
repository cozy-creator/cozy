//! The pinned host tier on the real code path: a real TensorFS store in a temp dir, a real
//! read lease and plan, a host-only plane (no GPU). Every check reads bytes back.

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
use tensorfs_plane::host::HostMem;
use tensorfs_plane::layout::{Layout, PART_ALIGN, REGION_ALIGN};
use tensorfs_plane::{Error, Plane, PlaneConfig, Tier, WsId};

const MIB: usize = 1 << 20;

/// (key, bytes): odd sizes on purpose; one tensor spans two 64 MiB objects, one is inline.
const TENSORS: &[(&str, usize)] = &[
    ("a.weight", 3 * MIB + 100),
    ("a.bias", 200),
    ("b.weight", 70 * MIB + 12),
    ("c.weight", MIB + 3),
    ("c.norm", 1000),
];

fn regions() -> Vec<Vec<String>> {
    [&["a.weight", "a.bias"][..], &["b.weight"], &["c.weight", "c.norm"]]
        .iter()
        .map(|r| r.iter().map(|k| format!("unet/{k}")).collect())
        .collect()
}

fn bytes_of(key: &str, n: usize) -> Vec<u8> {
    let seed = key.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    (0..n).map(|i| (seed.wrapping_add(i as u32).wrapping_mul(2654435761) >> 13) as u8).collect()
}

struct Fixture {
    root: PathBuf,
    store: Arc<Store>,
    meta: Arc<Meta>,
    header: Header,
    objects: Vec<ObjectRef>,
}

impl Fixture {
    fn new(name: &str) -> Fixture {
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

    fn plan(&self) -> ReadPlan {
        let traversal: Vec<(String, String)> =
            TENSORS.iter().map(|(k, _)| ("unet".into(), k.to_string())).collect();
        plan_for_traversal(&self.header, &traversal, &["unet".into()], 16 << 20).unwrap()
    }

    fn register(&self, plane: &Plane, host_fd: Option<i32>) -> WsId {
        let (lease, _) = read::acquire(&self.store, &self.meta, "test", self.objects.clone()).unwrap();
        plane
            .register("unet", self.store.clone(), self.meta.clone(), lease, &self.plan(), &regions(), host_fd)
            .unwrap()
    }

    /// Evict our own blob files from the page cache so a fill has to go to disk.
    fn drop_cache(&self) {
        for o in &self.objects {
            let f = std::fs::File::open(self.store.blob_path(&o.sha256)).unwrap();
            // SAFETY: advice on our own descriptor.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// Page-cache pages resident across our blob files (mincore; never faults them in).
    fn cached_pages(&self) -> usize {
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

fn host_only() -> Plane {
    Plane::open(PlaneConfig {
        readers: 2,
        ..Default::default()
    })
    .unwrap()
}

/// Every part's bytes in the memfd equal the source tensor's bytes.
fn check_bytes(plane: &Plane, ws: WsId, region: Option<u32>) {
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

#[test]
fn layout_aligns_every_tier_and_partitions_exactly() {
    let fx = Fixture::new("layout");
    let l = Layout::build(&fx.plan(), &regions()).unwrap();
    assert_eq!(l.regions.len(), 3);
    for r in &l.regions {
        assert_eq!(r.offset % REGION_ALIGN, 0);
        assert_eq!(r.span % REGION_ALIGN, 0);
        assert!(r.nbytes <= r.span);
    }
    let mut prev_end = 0;
    for p in &l.parts {
        assert_eq!(p.offset % PART_ALIGN, 0, "{}", p.what);
        let r = &l.regions[p.region as usize];
        assert!(p.offset >= r.offset && p.offset + p.nbytes <= r.offset + r.nbytes);
        if p.region == l.parts[0].region || p.offset >= prev_end {
            assert!(p.offset >= prev_end || p.offset == r.offset);
        }
        prev_end = p.offset + p.nbytes;
    }
    assert_eq!(l.nbytes, l.regions.last().map(|r| r.offset + r.span).unwrap());
    // The same inputs give the same digest; a different grouping does not.
    assert_eq!(Layout::build(&fx.plan(), &regions()).unwrap().digest, l.digest);
    let mut other = regions();
    let moved = other[2].pop().unwrap();
    other[0].push(moved);
    assert_ne!(Layout::build(&fx.plan(), &other).unwrap().digest, l.digest);

    let mut missing = regions();
    missing[1].clear();
    assert!(matches!(Layout::build(&fx.plan(), &missing), Err(Error::Invalid(_))));
    let mut twice = regions();
    twice[1].push("unet/a.bias".into());
    assert!(matches!(Layout::build(&fx.plan(), &twice), Err(Error::Invalid(_))));
    let mut stranger = regions();
    stranger[1].push("unet/nope".into());
    assert!(matches!(Layout::build(&fx.plan(), &stranger), Err(Error::Invalid(_))));
}

#[test]
fn cold_fill_goes_direct_and_leaves_no_page_cache_copy() {
    let fx = Fixture::new("cold");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    fx.drop_cache();
    let before = fx.cached_pages();
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, None);
    let s = plane.stats();
    let payload: u64 = TENSORS.iter().map(|(_, n)| *n as u64).sum();
    assert_eq!(
        s.host.direct_bytes + s.host.cached_bytes + s.host.buffered_bytes + s.host.inline_bytes,
        payload
    );
    assert!(s.host.direct_bytes > payload / 2, "cold bytes should come by O_DIRECT: {:?}", s.host);
    assert!(fx.cached_pages() <= before + 64, "the fill must not populate the page cache");
    assert_eq!(s.host.counters.fills, 3);
    assert!(s.sets[0].host_memfd_bytes >= s.host.ready_bytes);
    plane.close().unwrap();
}

#[test]
fn warm_fill_copies_from_the_page_cache() {
    let fx = Fixture::new("warm");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    // Warm the cache through ordinary reads.
    for o in &fx.objects {
        std::fs::read(fx.store.blob_path(&o.sha256)).unwrap();
    }
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, None);
    let s = plane.stats();
    assert!(s.host.cached_bytes > s.host.direct_bytes, "{:?}", s.host);
    plane.close().unwrap();
}

#[test]
fn budget_evicts_by_priority_then_tail_first_and_frees_ram() {
    let fx = Fixture::new("budget");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    let spans: Vec<u64> = plane.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    let total: u64 = spans.iter().sum();
    plane.set_pinned_budget(total).unwrap();
    plane.want(ws, Tier::Pinned, Some(&[0]), 5, false).unwrap().wait().unwrap();
    plane.want(ws, Tier::Pinned, Some(&[1, 2]), 1, false).unwrap().wait().unwrap();
    let ram_full = plane.stats().sets[0].host_memfd_bytes;

    // Equal priority 1: the tail (region 2) goes first.
    let t = plane.set_pinned_budget(total - spans[2]).unwrap();
    assert_eq!(t.evicted, vec![("unet".to_string(), 2)]);
    assert_eq!(t.over_budget_unreleasable, 0);
    let t = plane.set_pinned_budget(spans[0]).unwrap();
    assert_eq!(t.evicted, vec![("unet".to_string(), 1)]);
    let s = plane.stats();
    assert_eq!(s.host.used, spans[0]);
    assert!(s.sets[0].host_memfd_bytes < ram_full / 4, "punched regions must free RAM");
    check_bytes(&plane, ws, Some(0));

    // A want may not evict an equal or higher priority.
    plane.set_pinned_budget(spans[1]).unwrap();
    match plane.want(ws, Tier::Pinned, Some(&[1]), 5, false) {
        Err(Error::Shortfall { pool: "pinned", .. }) => {}
        Err(e) => panic!("expected Shortfall, got {e}"),
        Ok(_) => panic!("expected Shortfall"),
    }
    // A strictly higher priority may.
    plane.want(ws, Tier::Pinned, Some(&[1]), 9, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, Some(1));
    let s = plane.stats();
    assert_eq!(s.host.used, spans[1]);
    assert_eq!(s.host.counters.evictions, 3);

    // Pinned regions are never evicted implicitly; drop is explicit.
    plane.set_pinned_budget(total).unwrap();
    plane.want(ws, Tier::Pinned, Some(&[2]), 0, true).unwrap().wait().unwrap();
    let t = plane.set_pinned_budget(0).unwrap();
    assert_eq!(t.over_budget_unreleasable, spans[2]);
    assert_eq!(plane.drop_regions(ws, Tier::Pinned, Some(&[2])).unwrap(), spans[2]);
    plane.close().unwrap();
}

#[test]
fn a_second_description_adopts_ready_bytes_without_reading() {
    let fx = Fixture::new("adopt");
    let a = host_only();
    let wa = fx.register(&a, None);
    a.set_pinned_budget(1 << 30).unwrap();
    a.want(wa, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();

    let b = host_only();
    let wb = fx.register(&b, Some(a.host_fd(wa).unwrap()));
    b.set_pinned_budget(1 << 30).unwrap();
    b.want(wb, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    let s = b.stats();
    assert_eq!(s.host.counters.fill_bytes, 0, "adoption must not read the store");
    assert_eq!(s.host.direct_bytes + s.host.cached_bytes + s.host.buffered_bytes, 0);
    assert!(b.events().iter().all(|e| e.kind != "fill_done" || e.detail == "pinned<-adopted"));
    check_bytes(&b, wb, None);

    // B's punch clears the shared Ready word: A's next want refills.
    b.drop_regions(wb, Tier::Pinned, Some(&[1])).unwrap();
    a.drop_regions(wa, Tier::Pinned, Some(&[1])).unwrap();
    a.want(wa, Tier::Pinned, Some(&[1]), 0, false).unwrap().wait().unwrap();
    assert!(a.stats().host.counters.fill_bytes > 70 << 20);
    check_bytes(&a, wa, Some(1));
    b.close().unwrap();
    a.close().unwrap();
}

#[test]
fn a_punch_is_refused_while_another_description_fills() {
    let fx = Fixture::new("ofd");
    let l = Layout::build(&fx.plan(), &regions()).unwrap();
    let a = HostMem::create("ofd", &l).unwrap();
    let b = HostMem::adopt(a.fd(), &l).unwrap();
    a.fill_lock(1).unwrap();
    assert!(!b.punch(1, l.regions[1].offset, l.regions[1].span).unwrap());
    a.unlock(1);
    assert!(b.punch(1, l.regions[1].offset, l.regions[1].span).unwrap());
    // A layout mismatch is refused, not reinterpreted.
    let mut other = regions();
    let moved = other[2].pop().unwrap();
    other[0].push(moved);
    let l2 = Layout::build(&fx.plan(), &other).unwrap();
    assert!(HostMem::adopt(a.fd(), &l2).is_err());
}

#[test]
fn close_refuses_twice_and_releases_the_lease() {
    let fx = Fixture::new("close");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    plane.close_ws(ws).unwrap();
    assert!(matches!(plane.close_ws(ws), Err(Error::Closed)));
    assert_eq!(plane.stats().host.used, 0);
    // The store's hold is gone: a fresh lease and GC-free close work.
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "after", fx.objects.clone()).unwrap();
    lease.release(&fx.meta).unwrap();
    plane.close().unwrap();
}
