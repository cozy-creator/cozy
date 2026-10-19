//! SHA-256, the object identity: the `sha2` crate, which dispatches to the CPU's SHA
//! extensions (x86 SHA-NI, ARMv8 SHA2) at run time and falls back to portable code.

use sha2::Digest;

#[derive(Clone, Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

pub fn digest(data: &[u8]) -> [u8; 32] {
    sha2::Sha256::digest(data).into()
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

    /// FIPS 180-4's own vectors, one-shot and split across updates.
    #[test]
    fn known_answers() {
        let cases: [(&[u8], &str); 5] = [
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
            (
                &[b'a'; 1_000_000],
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            ),
        ];
        for (input, want) in cases {
            assert_eq!(hex_digest(input), want);
            let mut split = Sha256::new();
            for chunk in input.chunks(63) {
                split.update(chunk);
            }
            assert_eq!(hex(&split.finish()), want);
        }
    }
}
