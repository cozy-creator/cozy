//! CPU: regions of whole chunks, the chunk-aligned layout built from them, and sealed host
//! layouts adopted only when they hold exactly that layout.
#![cfg(target_os = "linux")]

mod common;

use std::os::fd::AsRawFd;

use common::*;
use tensorfs_core::client::regions;
use varena::host::{open_sealed, HostMem};
use varena::layout::{Layout, PART_ALIGN, REGION_ALIGN};
use varena::Error;

#[test]
fn regions_are_whole_chunks_split_at_size_and_boundaries() {
    let fx = Fixture::new("regions");
    let h = fx.ck.header();
    assert_eq!(regions(h, 64 << 20, &[]), vec![vec![0], vec![1, 2], vec![3], vec![4]], "blocks.1.w (70 MiB) is a region alone");
    assert_eq!(regions(h, 128 << 20, &[]), vec![vec![0, 1, 2, 3], vec![4]], "components never share a region");
    let blocks: Vec<String> = (0..4).map(|i| format!("unet/blocks.{i}")).collect();
    assert_eq!(regions(h, 128 << 20, &blocks), vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    // Chunk 3 holds blocks.2, blocks.3 and norm: one chunk is never split, whatever the boundaries.
    assert_eq!(regions(h, 1, &blocks), vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    // A prefix is a module path: "blocks.1" names neither "blocks.10" nor "blocks.1x".
    assert_eq!(regions(h, 128 << 20, &["unet/blocks.1".into()]), vec![vec![0], vec![1, 2], vec![3], vec![4]]);
    assert_eq!(regions(h, 128 << 20, &["unet/blocks.".into(), "vae/conv".into()]), vec![vec![0, 1, 2, 3], vec![4]]);
}

#[test]
fn the_layout_places_whole_chunks_and_every_part_once() {
    let fx = Fixture::new("layout");
    let l = Layout::build(fx.ck.header(), &fx.regions()).unwrap();
    for r in &l.regions {
        assert_eq!(r.offset % REGION_ALIGN, 0);
        assert_eq!(r.span % REGION_ALIGN, 0);
        assert!(r.chunks.iter().all(|c| c.offset % PART_ALIGN == 0));
    }
    assert_eq!(l.nbytes, l.regions.iter().map(|r| r.span).sum::<u64>());
    let named: Vec<String> = l.parts.iter().map(|p| p.what()).collect();
    let mut want: Vec<String> = TENSORS.iter().filter(|t| t.2 > 0).map(|t| format!("{}/{}", t.0, t.1)).collect();
    want.sort();
    let mut got = named.clone();
    got.sort();
    assert_eq!(got, want, "every non-empty tensor exactly once; the empty one has no bytes");
    // blocks.1.w runs from chunk 1 straight into chunk 2.
    let w = l.parts.iter().find(|p| p.key == "blocks.1.w").unwrap();
    let r = &l.regions[w.region as usize];
    assert_eq!((r.chunks.len(), r.chunks[1].offset), (2, r.chunks[0].offset + (64 << 20)));
    assert_eq!(w.nbytes as usize, size_of("unet", "blocks.1.w"));
    // The same grouping is the same digest; another grouping is not.
    assert_eq!(Layout::build(fx.ck.header(), &fx.regions()).unwrap().digest, l.digest);
    assert_ne!(Layout::build(fx.ck.header(), &[vec![0, 1, 2, 3], vec![4]]).unwrap().digest, l.digest);
    // A subset is a weight set of its own tensors.
    let sub = Layout::build(fx.ck.header(), &[vec![1, 2]]).unwrap();
    assert_eq!(sub.parts.iter().map(|p| p.key.as_str()).collect::<Vec<_>>(), ["blocks.1.w"]);
}

#[test]
fn a_tensor_split_across_regions_or_a_chunk_named_twice_is_refused() {
    let fx = Fixture::new("refused");
    let h = fx.ck.header();
    for bad in [vec![vec![1]], vec![vec![1], vec![2]], vec![vec![2, 1]], vec![vec![0], vec![0]], vec![vec![]], vec![vec![9]]] {
        assert!(matches!(Layout::build(h, &bad), Err(Error::Invalid(_))), "{bad:?}");
    }
}

#[test]
fn a_sealed_layout_is_adopted_only_for_its_own_layout() {
    let fx = Fixture::new("sealed");
    let l = Layout::build(fx.ck.header(), &fx.regions()).unwrap();
    let (fd, fill) = sealed(&fx, &l, Some(&[0, 1, 2]));
    drop(fill);
    let host = HostMem::adopt_sealed(fd.as_raw_fd(), &l).unwrap();
    assert!((0..3).all(|r| host.ready(r)));
    assert!(host.hole(3), "a region the owner does not hold is a hole");
    for r in 0..3u32 {
        let region = &l.regions[r as usize];
        // SAFETY: inside the adopted read-only mapping.
        let got = unsafe { std::slice::from_raw_parts(host.base().add(region.offset as usize), region.nbytes as usize) };
        check_region(&l, r, got);
    }
    // Another grouping of the same chunks is another layout.
    let other = Layout::build(fx.ck.header(), &[vec![0, 1, 2, 3], vec![4]]).unwrap();
    assert!(matches!(HostMem::adopt_sealed(fd.as_raw_fd(), &other), Err(Error::Invalid(_))));
    // A region left unfinished when its owner gives up is failed: no adopter waits for it.
    let (fd2, fill2) = open_sealed("unfinished", &l, None).unwrap();
    fill2.set_ready(0);
    drop(fill2);
    assert!(matches!(HostMem::adopt_sealed(fd2.as_raw_fd(), &l), Err(Error::Invalid(_))));
    // Sealed: no new writable mapping of the handed-out fd.
    // SAFETY: an mmap attempt on our own fd; a refusal returns MAP_FAILED.
    let p = unsafe { libc::mmap(std::ptr::null_mut(), 4096, libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
    assert_eq!(p, libc::MAP_FAILED, "the owner's seals must refuse a writable mapping");
}
