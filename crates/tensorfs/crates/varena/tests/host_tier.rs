//! The host tier on the real code path: a real TensorFS store in a temp dir, a real plan, and
//! the memfd its layout lives in. Every check reads bytes back or asks the kernel.
#![cfg(target_os = "linux")] // The host tier (memfd, seals, OFD locks) is Linux-only.

mod common;

use common::*;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use varena::host::{hold, release_hold, HostMem};
use varena::layout::{Layout, PART_ALIGN, REGION_ALIGN};
use varena::Error;

fn dup(fd: i32) -> OwnedFd {
    // SAFETY: duplicating a live descriptor; the copy is ours.
    unsafe { OwnedFd::from_raw_fd(libc::dup(fd)) }
}

fn resident(fd: i32) -> u64 {
    // SAFETY: fstat into a zeroed struct.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut st) }, 0);
    st.st_blocks as u64 * 512
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
