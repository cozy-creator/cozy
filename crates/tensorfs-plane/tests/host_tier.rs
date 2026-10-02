//! The pinned host tier on the real code path: a real TensorFS store in a temp dir, a real
//! read lease and plan, a host-only plane (no GPU). Every check reads bytes back.

mod common;

use common::*;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::fit::{fit, Custody, Fit, TensorRequirements};
use tensorfs_core::read;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use tensorfs_plane::host::{hold, release_hold, HostMem};
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
fn a_release_punches_only_for_the_last_claimant() {
    let fx = Fixture::new("ofd");
    let l = Layout::build(&fx.plan(), &regions()).unwrap();
    let (off, span) = (l.regions[1].offset, l.regions[1].span);
    let a = HostMem::create("ofd", &l).unwrap();
    let b = HostMem::adopt(a.fd(), &l).unwrap();
    // SAFETY: region 1 of our own mapping.
    unsafe { std::ptr::write_bytes(a.base().add(off as usize), 7, span as usize) };
    a.set_ready(1, true);
    let filled = resident(a.fd());
    a.claim(1).unwrap();
    b.claim(1).unwrap();
    assert!(!b.release(1, off, span).unwrap(), "A still claims the region: nothing is punched");
    assert!(a.ready(1));
    assert_eq!(resident(a.fd()), filled);
    // The descriptor handed to other processes holds no claim: keeping it blocks no punch.
    let kept = dup(a.fd());
    assert!(a.release(1, off, span).unwrap(), "the last claimant punches");
    assert!(!b.ready(1));
    assert!(resident(kept.as_raw_fd()) < filled - span / 2);
    // A whole-tier hold (the Worker's) keeps every region from the punch; releasing it punches
    // the Ready regions no description claims, and only those.
    let fill = |r: usize| {
        let (o, n) = (l.regions[r].offset, l.regions[r].span);
        // SAFETY: region r of our own mapping.
        unsafe { std::ptr::write_bytes(a.base().add(o as usize), 9, n as usize) };
        a.set_ready(r as u32, true);
    };
    fill(1);
    fill(2);
    let both = resident(kept.as_raw_fd());
    a.claim(1).unwrap();
    a.claim(2).unwrap();
    let held = hold(kept.as_raw_fd()).unwrap();
    assert!(!a.release(1, off, span).unwrap(), "the hold still claims region 1");
    assert!(a.ready(1) && resident(kept.as_raw_fd()) == both);
    assert_eq!(release_hold(held).unwrap(), span, "only the region A let go is punched");
    assert!(!a.ready(1) && a.ready(2));
    assert!(resident(kept.as_raw_fd()) <= both - span);
    assert!(a.release(2, l.regions[2].offset, l.regions[2].span).unwrap());
    // A layout mismatch is refused, not reinterpreted.
    let mut other = regions();
    let moved = other[2].pop().unwrap();
    other[0].push(moved);
    let l2 = Layout::build(&fx.plan(), &other).unwrap();
    assert!(HostMem::adopt(a.fd(), &l2).is_err());
}

#[test]
fn closing_a_weight_set_frees_its_pinned_ram_while_a_handle_survives() {
    // The Worker keeps every offered memfd (HostTiers): a closed weight set's pages must not
    // stay resident outside every budget.
    let fx = Fixture::new("closefree");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    let kept = dup(plane.host_fd(ws).unwrap());
    let empty = resident(kept.as_raw_fd());
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    assert!(resident(kept.as_raw_fd()) > 70 << 20);
    plane.close_ws(ws).unwrap();
    assert_eq!(resident(kept.as_raw_fd()), empty, "close must punch what no one else claims");
    plane.close().unwrap();
}

#[test]
fn an_adopter_counts_ready_bytes_and_the_last_claimant_frees_them() {
    let fx = Fixture::new("claims");
    let a = host_only();
    let wa = fx.register(&a, None);
    a.set_pinned_budget(1 << 30).unwrap();
    a.want(wa, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    let total: u64 = a.layout(wa).unwrap().regions.iter().map(|r| r.span).sum();
    let kept = dup(a.host_fd(wa).unwrap());
    let full = resident(kept.as_raw_fd());

    // B adopts the memfd: what it already holds is in B's books at once.
    let b = host_only();
    let wb = fx.register(&b, Some(a.host_fd(wa).unwrap()));
    assert_eq!(b.stats().host.used, total, "adopted RAM must count against the adopter's budget");
    b.set_pinned_budget(1 << 30).unwrap();
    b.want(wb, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    assert_eq!(b.stats().host.counters.fill_bytes, 0);

    // A lets go of everything: B still claims it, so no page leaves under B.
    a.close_ws(wa).unwrap();
    assert_eq!(resident(kept.as_raw_fd()), full);
    check_bytes(&b, wb, None);
    // A budget below what B holds evicts the adopted regions first: B is the last claimant.
    let t = b.set_pinned_budget(0).unwrap();
    assert_eq!(t.freed, total);
    assert!(resident(kept.as_raw_fd()) < full / 8, "the last claimant's eviction frees the RAM");
    b.close().unwrap();
    a.close().unwrap();
}

#[test]
fn a_failed_fill_holds_no_ram_and_no_budget() {
    let fx = Fixture::new("failed");
    let plane = host_only();
    let ws = fx.register(&plane, None);
    plane.set_pinned_budget(1 << 30).unwrap();
    let spans: Vec<u64> = plane.layout(ws).unwrap().regions.iter().map(|r| r.span).collect();
    // b.weight spans two objects: lose the second, so the first lands and the fill still fails.
    let tail = fx.objects.iter().find(|o| o.length == (6 * MIB + 12) as u64).unwrap();
    std::fs::remove_file(fx.store.blob_path(&tail.sha256)).unwrap();
    let kept = dup(plane.host_fd(ws).unwrap());
    let empty = resident(kept.as_raw_fd());
    assert!(plane.want(ws, Tier::Pinned, Some(&[1]), 0, false).unwrap().wait().is_err());
    assert_eq!(plane.stats().host.used, 0);
    assert_eq!(resident(kept.as_raw_fd()), empty, "a failed fill's pages must be punched");
    // The failed region frees nothing more and the books stay whole.
    assert_eq!(plane.drop_regions(ws, Tier::Pinned, Some(&[1])).unwrap(), 0);
    plane.want(ws, Tier::Pinned, Some(&[0, 2]), 0, false).unwrap().wait().unwrap();
    assert_eq!(plane.stats().host.used, spans[0] + spans[2]);
    plane.close_ws(ws).unwrap();
    assert_eq!(resident(kept.as_raw_fd()), empty);
    plane.close().unwrap();
}

/// A stored tensor the code does not build (`fit` skips it with a warning) is in no plan and
/// no layout. With its objects gone from the store the fill still lands every built byte:
/// nothing read it, and nothing mapped room for it.
#[test]
fn a_tensor_fit_skips_is_never_read_or_mapped() {
    let fx = Fixture::new("skipped");
    let built: Vec<(&str, usize)> = TENSORS.iter().copied().filter(|(k, _)| *k != "b.weight").collect();
    let requirements = TensorRequirements::new(
        built.iter().map(|(k, n)| ("unet".to_string(), k.to_string(), vec![*n as u64], Some(Dtype::U8))),
    )
    .unwrap();
    let skipped = (70 * MIB + 12) as u64;
    let verdict = fit(&requirements, &fx.header, Custody::Canonical, false, None, None).unwrap();
    let Fit::Ok { ignored, ignored_bytes, .. } = &verdict else {
        panic!("{}", verdict.text());
    };
    assert_eq!((ignored.as_slice(), *ignored_bytes), (&["unet/b.weight".to_string()][..], skipped));
    assert_eq!(
        verdict.warning().unwrap(),
        format!("1 stored tensor(s) the code does not build were skipped, not loaded: {skipped} B (unet/b.weight)")
    );

    // The caller's plan names every stored tensor and keeps the parts it builds.
    let whats: Vec<String> = built.iter().map(|(k, _)| format!("unet/{k}#value")).collect();
    let plan = fx.plan().select(&whats).unwrap();
    let regions: Vec<Vec<String>> = regions().into_iter().filter(|r| r != &["unet/b.weight".to_string()]).collect();
    let (lease, _) = read::acquire(&fx.store, &fx.meta, "test", fx.objects.clone()).unwrap();
    let plane = host_only();
    let source = plane.source(fx.store.clone(), fx.meta.clone(), lease);
    let ws = plane.register("unet", source, &plan, &regions, None).unwrap();
    // A fill that needs an object that is gone fails (`a_failed_fill_holds_no_ram_and_no_budget`).
    for object in fx.objects.iter().filter(|o| o.length >= (6 * MIB) as u64) {
        std::fs::remove_file(fx.store.blob_path(&object.sha256)).unwrap();
    }
    plane.set_pinned_budget(1 << 30).unwrap();
    plane.want(ws, Tier::Pinned, None, 0, false).unwrap().wait().unwrap();
    check_bytes(&plane, ws, None);

    let layout = plane.layout(ws).unwrap();
    assert!(layout.parts.iter().all(|p| !p.what.contains("b.weight")));
    assert!(layout.nbytes < skipped, "the pinned tier made room for the skipped tensor: {} B", layout.nbytes);
    let payload: u64 = built.iter().map(|(_, n)| *n as u64).sum();
    let host = plane.stats().host;
    assert_eq!(host.direct_bytes + host.cached_bytes + host.buffered_bytes + host.inline_bytes, payload);
    plane.close().unwrap();
}

fn dup(fd: i32) -> OwnedFd {
    // SAFETY: duplicating a live descriptor; the copy is ours.
    unsafe { OwnedFd::from_raw_fd(libc::dup(fd)) }
}

/// RAM the memfd behind `fd` holds (kernel truth).
fn resident(fd: i32) -> u64 {
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut st) }, 0);
    st.st_blocks as u64 * 512
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
