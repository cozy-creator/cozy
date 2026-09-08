//! sha256 — the one hash algorithm (tensorfs.md §7). No dependencies.
//!
//! A digest is identity here: `put_stream` is the single digest-enforced admission door,
//! an object id IS `sha256:<64 hex>`, and repo-CAS correctness rests on this file computing
//! exactly SHA-256. So the compression function exists three times — a portable scalar
//! routine, an x86-64 SHA-extension routine, an aarch64 one — and the three are required
//! to agree bit for bit on every input. `tests/sha256_backends.rs` is what fails if they
//! ever do not.
//!
//! Why hand-written intrinsics rather than a crate (tfs-068). `sha2` with its intrinsic
//! backends gives the same bytes, and was rejected because this file's freedom from
//! dependencies is a deliberate statement about the format — a reader of the spec can read
//! the hash — and it would put seven transitive crates (digest, block-buffer,
//! crypto-common, hybrid-array, typenum, cpufeatures, cfg-if) inside the one module whose
//! whole job is to be auditable. `sha2`'s `asm` feature was rejected separately: it wants a
//! C toolchain at build time, which is a wheel-build cost the Xet evaluation (tfs-061)
//! already declined on.
//!
//! That choice has one honest price, and it is charged in `lib.rs`: intrinsics are `unsafe`,
//! so the crate is `deny(unsafe_code)` rather than `forbid`. Taking the crate instead would
//! have kept `forbid` by moving the same `unsafe` behind someone else's name. If that trade
//! is ever re-decided, `sha2` is the replacement and this module's public surface —
//! `Sha256::new`, `Sha256::scalar`, `update`, `finish`, `digest`, `hex` — is what it has to
//! keep.
//!
//! Dispatch is at RUNTIME, once, because the wheel is built once and runs on whatever the
//! pod rents: a binary built on a host with the extensions must still be correct on a host
//! without them. `Sha256::scalar()` pins an instance to the portable routine so the
//! fallback is exercised on every machine, including CI, whatever CPU it happens to be.

use std::sync::OnceLock;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Which compression routine an instance runs. `Hardware` is only ever produced by
/// [`detect`], and only on a target whose instructions this file implements.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Scalar,
    Hardware,
}

/// What the process will use, in words a human can put in a bug report.
pub fn backend_name() -> &'static str {
    match backend() {
        Backend::Scalar => "scalar",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Hardware => "x86-64 sha extensions",
        #[cfg(target_arch = "aarch64")]
        Backend::Hardware => "aarch64 sha2 extensions",
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Backend::Hardware => "hardware",
    }
}

/// Detected once per process, never per call.
pub fn backend() -> Backend {
    static ONCE: OnceLock<Backend> = OnceLock::new();
    *ONCE.get_or_init(detect)
}

fn detect() -> Backend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // `sha256rnds2` needs SSE2; `_mm_shuffle_epi8` needs SSSE3 and `_mm_blend_epi16` SSE4.1.
    // No shipped CPU has the SHA extensions without all three, but ask anyway.
    if std::arch::is_x86_feature_detected!("sha")
        && std::arch::is_x86_feature_detected!("sse2")
        && std::arch::is_x86_feature_detected!("ssse3")
        && std::arch::is_x86_feature_detected!("sse4.1")
    {
        return Backend::Hardware;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("sha2") {
        return Backend::Hardware;
    }
    Backend::Scalar
}

/// `blocks.len()` is a nonzero multiple of 64.
// One of the crate's two `unsafe` sites (lib.rs): the call into an intrinsics routine.
#[allow(unsafe_code)]
fn compress_blocks(backend: Backend, h: &mut [u32; 8], blocks: &[u8]) {
    debug_assert!(!blocks.is_empty() && blocks.len().is_multiple_of(64));
    match backend {
        Backend::Scalar => compress_scalar(h, blocks),
        Backend::Hardware => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Backend::Hardware` exists only where `detect` found the features.
            unsafe {
                x86::compress(h, blocks)
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: as above.
            unsafe {
                arm::compress(h, blocks)
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            compress_scalar(h, blocks)
        }
    }
}

fn compress_scalar(h: &mut [u32; 8], blocks: &[u8]) {
    let mut w = [0u32; 64];
    for block in blocks.chunks_exact(64) {
        for (i, word) in w[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = *h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip(v.iter()) {
            *x = x.wrapping_add(*y);
        }
    }
}

// ---------------------------------------------------------------- x86-64 SHA extensions

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86 {
    use super::K;
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// State is carried as ABEF / CDGH, which is the order `sha256rnds2` wants; the two
    /// shuffles at entry and exit are the whole cost of that convention. Four message
    /// words at a time, two rounds per `sha256rnds2`.
    ///
    /// # Safety
    /// The caller has established `sha`, `sse2`, `ssse3` and `sse4.1`.
    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    pub unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
        // Big-endian word order out of a little-endian 16-byte load.
        let mask = _mm_set_epi64x(
            0x0c0d_0e0f_0809_0a0b_u64 as i64,
            0x0405_0607_0001_0203_u64 as i64,
        );

        let dcba = _mm_loadu_si128(state.as_ptr() as *const __m128i);
        let hgfe = _mm_loadu_si128(state.as_ptr().add(4) as *const __m128i);
        let cdab = _mm_shuffle_epi32(dcba, 0xb1);
        let efgh = _mm_shuffle_epi32(hgfe, 0x1b);
        let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
        let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xf0);

        // Two rounds, then two more: `sha256rnds2` takes the low two words of its message
        // operand, so the same value is fed twice with a shuffle between.
        macro_rules! rounds4 {
            ($abef:ident, $cdgh:ident, $w:expr, $i:expr) => {{
                let kv = _mm_set_epi32(
                    K[4 * $i + 3] as i32,
                    K[4 * $i + 2] as i32,
                    K[4 * $i + 1] as i32,
                    K[4 * $i] as i32,
                );
                let t = _mm_add_epi32($w, kv);
                $cdgh = _mm_sha256rnds2_epu32($cdgh, $abef, t);
                $abef = _mm_sha256rnds2_epu32($abef, $cdgh, _mm_shuffle_epi32(t, 0x0e));
            }};
        }
        // w[n..n+4] from the four preceding quads: msg1 is the σ0 half, msg2 the σ1 half.
        macro_rules! schedule {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {{
                let t = _mm_add_epi32(_mm_sha256msg1_epu32($a, $b), _mm_alignr_epi8($d, $c, 4));
                $a = _mm_sha256msg2_epu32(t, $d);
            }};
        }

        for block in blocks.chunks_exact(64) {
            let p = block.as_ptr() as *const __m128i;
            let abef_in = abef;
            let cdgh_in = cdgh;

            let mut w0 = _mm_shuffle_epi8(_mm_loadu_si128(p), mask);
            let mut w1 = _mm_shuffle_epi8(_mm_loadu_si128(p.add(1)), mask);
            let mut w2 = _mm_shuffle_epi8(_mm_loadu_si128(p.add(2)), mask);
            let mut w3 = _mm_shuffle_epi8(_mm_loadu_si128(p.add(3)), mask);

            rounds4!(abef, cdgh, w0, 0);
            rounds4!(abef, cdgh, w1, 1);
            rounds4!(abef, cdgh, w2, 2);
            rounds4!(abef, cdgh, w3, 3);
            schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w0, 4);
            schedule!(w1, w2, w3, w0);
            rounds4!(abef, cdgh, w1, 5);
            schedule!(w2, w3, w0, w1);
            rounds4!(abef, cdgh, w2, 6);
            schedule!(w3, w0, w1, w2);
            rounds4!(abef, cdgh, w3, 7);
            schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w0, 8);
            schedule!(w1, w2, w3, w0);
            rounds4!(abef, cdgh, w1, 9);
            schedule!(w2, w3, w0, w1);
            rounds4!(abef, cdgh, w2, 10);
            schedule!(w3, w0, w1, w2);
            rounds4!(abef, cdgh, w3, 11);
            schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w0, 12);
            schedule!(w1, w2, w3, w0);
            rounds4!(abef, cdgh, w1, 13);
            schedule!(w2, w3, w0, w1);
            rounds4!(abef, cdgh, w2, 14);
            schedule!(w3, w0, w1, w2);
            rounds4!(abef, cdgh, w3, 15);

            abef = _mm_add_epi32(abef, abef_in);
            cdgh = _mm_add_epi32(cdgh, cdgh_in);
        }

        let feba = _mm_shuffle_epi32(abef, 0x1b);
        let dchg = _mm_shuffle_epi32(cdgh, 0xb1);
        _mm_storeu_si128(
            state.as_mut_ptr() as *mut __m128i,
            _mm_blend_epi16(feba, dchg, 0xf0),
        );
        _mm_storeu_si128(
            state.as_mut_ptr().add(4) as *mut __m128i,
            _mm_alignr_epi8(dchg, feba, 8),
        );
    }
}

// ---------------------------------------------------------------- aarch64 SHA2 extensions

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod arm {
    use super::K;
    use core::arch::aarch64::*;

    /// `sha256h`/`sha256h2` are the two halves of four rounds; `sha256su0`/`sha256su1` are
    /// the two halves of the message schedule. State stays ABCD / EFGH here — unlike x86,
    /// no entry shuffle is needed.
    ///
    /// # Safety
    /// The caller has established `sha2` (which implies `neon`).
    #[target_feature(enable = "neon,sha2")]
    pub unsafe fn compress(state: &mut [u32; 8], blocks: &[u8]) {
        // A local copy: `K` is a `const`, so `K.as_ptr()` would name a temporary.
        let k: [u32; 64] = K;
        let mut abcd = vld1q_u32(state.as_ptr());
        let mut efgh = vld1q_u32(state.as_ptr().add(4));

        // Four rounds against a pre-added w+k. `sha256h2` needs the PREVIOUS abcd, so it
        // is saved before `sha256h` overwrites it.
        macro_rules! rounds4 {
            ($abcd:ident, $efgh:ident, $w:expr, $i:expr) => {{
                let wk = vaddq_u32($w, vld1q_u32(k.as_ptr().add(4 * $i)));
                let prev = $abcd;
                $abcd = vsha256hq_u32($abcd, $efgh, wk);
                $efgh = vsha256h2q_u32($efgh, prev, wk);
            }};
        }
        // w[n..n+4] from the four preceding quads: su0 is the σ0 half, su1 the σ1 half.
        macro_rules! schedule {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {{
                $a = vsha256su1q_u32(vsha256su0q_u32($a, $b), $c, $d);
            }};
        }

        for block in blocks.chunks_exact(64) {
            let abcd_in = abcd;
            let efgh_in = efgh;

            let p = block.as_ptr();
            let mut w0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p)));
            let mut w1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(16))));
            let mut w2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(32))));
            let mut w3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p.add(48))));

            rounds4!(abcd, efgh, w0, 0);
            schedule!(w0, w1, w2, w3);
            rounds4!(abcd, efgh, w1, 1);
            schedule!(w1, w2, w3, w0);
            rounds4!(abcd, efgh, w2, 2);
            schedule!(w2, w3, w0, w1);
            rounds4!(abcd, efgh, w3, 3);
            schedule!(w3, w0, w1, w2);

            rounds4!(abcd, efgh, w0, 4);
            schedule!(w0, w1, w2, w3);
            rounds4!(abcd, efgh, w1, 5);
            schedule!(w1, w2, w3, w0);
            rounds4!(abcd, efgh, w2, 6);
            schedule!(w2, w3, w0, w1);
            rounds4!(abcd, efgh, w3, 7);
            schedule!(w3, w0, w1, w2);

            rounds4!(abcd, efgh, w0, 8);
            schedule!(w0, w1, w2, w3);
            rounds4!(abcd, efgh, w1, 9);
            schedule!(w1, w2, w3, w0);
            rounds4!(abcd, efgh, w2, 10);
            schedule!(w2, w3, w0, w1);
            rounds4!(abcd, efgh, w3, 11);
            schedule!(w3, w0, w1, w2);

            // The schedule is complete; the last four quads are rounds only.
            rounds4!(abcd, efgh, w0, 12);
            rounds4!(abcd, efgh, w1, 13);
            rounds4!(abcd, efgh, w2, 14);
            rounds4!(abcd, efgh, w3, 15);

            abcd = vaddq_u32(abcd, abcd_in);
            efgh = vaddq_u32(efgh, efgh_in);
        }

        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}

// ---------------------------------------------------------------- the hasher

#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    len: usize,
    total: u64,
    backend: Backend,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// The process's fastest correct routine.
    pub fn new() -> Self {
        Self::with(backend())
    }

    /// The portable routine, whatever the CPU offers. This is what the differential test
    /// compares against, and the reason the fallback is exercised on every machine.
    pub fn scalar() -> Self {
        Self::with(Backend::Scalar)
    }

    fn with(backend: Backend) -> Self {
        Sha256 {
            h: IV,
            buf: [0u8; 64],
            len: 0,
            total: 0,
            backend,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.len != 0 {
            let take = core::cmp::min(64 - self.len, data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len < 64 {
                return; // `data` is necessarily empty here
            }
            compress_blocks(self.backend, &mut self.h, &self.buf);
            self.len = 0;
        }
        // The bulk: whole blocks straight out of the caller's buffer, never copied.
        let full = data.len() & !63;
        if full != 0 {
            compress_blocks(self.backend, &mut self.h, &data[..full]);
        }
        let rest = &data[full..];
        self.buf[..rest.len()].copy_from_slice(rest);
        self.len = rest.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.total.wrapping_mul(8);
        let len = self.len;
        self.buf[len] = 0x80;
        self.buf[len + 1..].fill(0);
        if len >= 56 {
            // No room for the 8-byte length: close this block and pad into a fresh one.
            compress_blocks(self.backend, &mut self.h, &self.buf);
            self.buf = [0u8; 64];
        }
        self.buf[56..64].copy_from_slice(&bits.to_be_bytes());
        compress_blocks(self.backend, &mut self.h, &self.buf);
        let mut out = [0u8; 32];
        for (i, word) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

pub fn digest(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finish()
}

pub fn hex(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 0xf) as usize] as char);
    }
    s
}

pub fn hex_digest(data: &[u8]) -> String {
    hex(&digest(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4's own vectors, against the scalar routine and against whatever this CPU
    /// dispatches to. These anchor the differential test: agreement between two wrong
    /// implementations would prove nothing.
    #[test]
    fn known_answers() {
        let cases: [(&[u8], &str); 4] = [
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn\
                  hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (input, want) in cases {
            assert_eq!(hex_digest(input), want, "dispatched backend");
            let mut s = Sha256::scalar();
            s.update(input);
            assert_eq!(hex(&s.finish()), want, "scalar backend");
        }
    }

    #[test]
    fn a_million_a() {
        let data = vec![b'a'; 1_000_000];
        assert_eq!(
            hex_digest(&data),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn backend_is_stable() {
        assert_eq!(backend(), backend());
        assert!(!backend_name().is_empty());
    }
}
