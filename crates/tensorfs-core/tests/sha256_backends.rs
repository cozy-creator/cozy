//! The two SHA-256 routines are one hash function or they are silent corruption of every
//! object id in the store (tfs-068). This is the test that fails if they ever disagree by
//! a single bit.
//!
//! `Sha256::new()` dispatches to the CPU's SHA extensions where they exist; `Sha256::scalar()`
//! is always the portable routine. Both arms run on every machine, so the fallback is
//! exercised even on a CI runner that has the extensions — and the differential half is
//! exercised on any machine that does.

use tensorfs_core::sha256::{self, Backend, Sha256};

/// Deterministic, dependency-free, and not a constant pattern: a constant input would let a
/// broken message schedule agree with a correct one.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            out.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        out.truncate(n);
        out
    }
}

fn dispatched(data: &[u8]) -> [u8; 32] {
    sha256::digest(data)
}

fn scalar(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::scalar();
    h.update(data);
    h.finish()
}

fn agree(data: &[u8], what: &str) {
    let a = dispatched(data);
    let b = scalar(data);
    assert_eq!(
        sha256::hex(&a),
        sha256::hex(&b),
        "{what}: {} backend disagrees with scalar at len {}",
        sha256::backend_name(),
        data.len()
    );
}

/// Every length from empty through four blocks plus change. Empty input is length 0 and is
/// therefore in this loop on purpose.
#[test]
fn every_length_to_256_agrees() {
    let mut rng = Rng::new(0x243f_6a88_85a3_08d3);
    for n in 0..=256usize {
        let data = rng.bytes(n);
        agree(&data, "sweep");
    }
}

/// The lengths where padding and blocking decide the answer: one short of a block, exactly a
/// block, one over; the 55/56 boundary where the length field stops fitting and a second
/// padding block appears; and the same shapes a block and two blocks further on.
#[test]
fn padding_and_block_boundaries_agree() {
    let mut rng = Rng::new(0x13198a2e_03707344);
    for n in [
        0usize, 1, 54, 55, 56, 57, 63, 64, 65, 111, 112, 118, 119, 120, 127, 128, 129, 191, 192,
        255, 256, 257,
    ] {
        let data = rng.bytes(n);
        agree(&data, "boundary");
    }
}

/// The store reads through a 1 MiB buffer (`store::BUF`), so the sizes either side of it are
/// where a bulk path that mishandles its tail would show up, and 64 MiB is the real segment
/// size the pipeline hashes at.
#[test]
fn buffer_sized_inputs_agree() {
    let mut rng = Rng::new(0xa409_3822_299f_31d0);
    let mib = 1usize << 20;
    for n in [mib - 65, mib - 1, mib, mib + 1, mib + 63, 4 * mib + 37] {
        let data = rng.bytes(n);
        agree(&data, "buffer");
    }
}

/// The same bytes fed in arbitrary pieces must give the same digest as one call — this is
/// what the store actually does, one `read` at a time, and it is the half of `update` that
/// the bulk path rewrote.
#[test]
fn arbitrary_chunking_agrees() {
    let mut rng = Rng::new(0x082e_fa98_ec4e_6c89);
    let data = rng.bytes(300_000);
    let want = scalar(&data);

    for trial in 0..64 {
        let mut split = Rng::new(0x4528_21e6_38d0_1377 + trial);
        for (label, mut h) in [("dispatched", Sha256::new()), ("scalar", Sha256::scalar())] {
            let mut rest = &data[..];
            while !rest.is_empty() {
                // Chunks that straddle the 64-byte block, sometimes tiny, sometimes long.
                let n = (split.next_u64() as usize % 200).min(rest.len());
                let n = if n == 0 { rest.len().min(1) } else { n };
                h.update(&rest[..n]);
                rest = &rest[n..];
            }
            assert_eq!(
                sha256::hex(&h.finish()),
                sha256::hex(&want),
                "chunked update, trial {trial}, {label}"
            );
        }
    }
}

/// A zero-length `update` must not disturb the buffer, and neither must a run of them.
#[test]
fn empty_updates_are_inert() {
    for (label, mut h) in [("dispatched", Sha256::new()), ("scalar", Sha256::scalar())] {
        h.update(b"");
        h.update(b"ab");
        h.update(b"");
        h.update(b"c");
        h.update(b"");
        assert_eq!(
            sha256::hex(&h.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "{label}"
        );
    }
}

/// FIPS 180-4 through the dispatched path. The differential tests prove the two routines
/// agree; this proves what they agree ON.
#[test]
fn known_answers_through_the_dispatched_path() {
    assert_eq!(
        sha256::hex_digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256::hex_digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256::hex_digest(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

/// Detection is a decision about correctness, not about speed: a CPU with the instructions
/// that dispatches to the scalar routine is a regression in this file's `detect`, and one
/// without them that dispatches to the intrinsics is an illegal instruction on a pod.
#[test]
fn dispatch_matches_the_cpu() {
    let hardware = sha256::backend() == Backend::Hardware;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert_eq!(
        hardware,
        std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1"),
        "dispatch disagrees with x86 feature detection"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        hardware,
        std::arch::is_aarch64_feature_detected!("sha2"),
        "dispatch disagrees with aarch64 feature detection"
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    assert!(!hardware, "no hardware routine exists for this target");
    // Recorded so a failure elsewhere in the suite can be read against the path that ran.
    eprintln!("sha256 backend: {}", sha256::backend_name());
}
