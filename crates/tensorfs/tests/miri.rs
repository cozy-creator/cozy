//! The read engine's host paths under Miri (xs-067): host-to-host copies cut on the request grid,
//! on one thread and several, and zero fills. `cargo +nightly miri test -p tensorfs
//! --no-default-features --test miri`; the same tests run natively with the crate's suites.

use tensorfs::io::{read, Copy, Dst, ReadOpts, Src};

fn opts(threads: usize) -> ReadOpts {
    ReadOpts { threads, request: 4096, streams: 1, direct: false }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
}

#[test]
fn host_copies_land_whole_on_one_thread_and_several() {
    for threads in [1, 3] {
        let sources = [pattern(10_000, 1), pattern(1, 2), pattern(4096, 3), pattern(0, 4)];
        let mut out: Vec<Vec<u8>> = sources.iter().map(|s| vec![0xAA; s.len()]).collect();
        let copies: Vec<Copy> = (sources.iter().zip(out.iter_mut()))
            .map(|(s, d)| Copy { src: Src::Host(s.as_ptr()), dst: Dst::host(d) })
            .collect();
        // SAFETY: every source and destination is a live, distinct Vec that outlives the call.
        let stats = unsafe { read(&copies, &opts(threads)) }.unwrap();
        assert_eq!(stats.bytes, 10_000 + 1 + 4096);
        assert_eq!(out, sources);
    }
}

#[test]
fn zero_fills_only_their_destination() {
    let mut buf = vec![7u8; 9000];
    let (head, rest) = buf.split_at_mut(100);
    let (middle, _) = rest.split_at_mut(8000);
    let copies = [Copy { src: Src::Zero, dst: Dst::host(middle) }, Copy { src: Src::Zero, dst: Dst::host(head) }];
    // SAFETY: two disjoint parts of one live buffer.
    unsafe { read(&copies, &opts(2)) }.unwrap();
    assert!(buf[..8100].iter().all(|&b| b == 0));
    assert!(buf[8100..].iter().all(|&b| b == 7));
}
