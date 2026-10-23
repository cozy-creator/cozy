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
