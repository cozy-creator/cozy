//! The pinned host tier on the real code path: a real TensorFS store in a temp dir, a real
//! read lease and plan, a host-only plane (no GPU). Every check reads bytes back.

mod common;

use common::*;
use tensorfs_core::read;
use tensorfs_plane::host::HostMem;
use tensorfs_plane::layout::{Layout, PART_ALIGN, REGION_ALIGN};
use tensorfs_plane::{Error, Tier};

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
    let before = fx.holds();
    plane.close_ws(ws).unwrap();
    assert_eq!(fx.holds(), before - 1, "the last weight set on a source ends its lease");
    assert!(matches!(plane.close_ws(ws), Err(Error::Closed)));
    assert_eq!(plane.stats().host.used, 0);
    // The store's hold is gone: a fresh lease and GC-free close work.
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "after", fx.objects.clone()).unwrap();
    lease.release(&fx.meta).unwrap();
    plane.close().unwrap();
}

#[test]
fn one_source_serves_several_weight_sets() {
    let fx = Fixture::new("source");
    let plane = host_only();
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "shared", fx.objects.clone()).unwrap();
    let src = plane.source(fx.store.clone(), fx.meta.clone(), lease);
    let base = fx.holds();
    // Two groupings (by tensor and by exact part name) of the same plan share one lease.
    let by_part: Vec<Vec<String>> = regions()
        .into_iter()
        .map(|r| r.into_iter().map(|t| format!("{t}#value")).collect())
        .collect();
    let a = plane.register("a", src.clone(), &fx.plan(), &regions(), None).unwrap();
    let b = plane.register("b", src.clone(), &fx.plan(), &by_part, None).unwrap();
    drop(src);
    assert_eq!(plane.layout(a).unwrap().digest, plane.layout(b).unwrap().digest);
    plane.set_pinned_budget(1 << 30).unwrap();
    plane.want(a, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    plane.want(b, Tier::Pinned, Some(&[1]), 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, a, None);
    check_bytes(&plane, b, Some(1));
    plane.close_ws(a).unwrap();
    assert_eq!(fx.holds(), base, "the lease lives while a weight set uses it");
    plane.close_ws(b).unwrap();
    assert_eq!(fx.holds(), base - 1);
    plane.close().unwrap();
}
