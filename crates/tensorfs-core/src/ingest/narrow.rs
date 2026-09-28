//! The one declared value change at the ingest border (`Converter::lane`): bf16/f32 source
//! elements narrow to f16. Bit-for-bit numpy's `astype(float16)` (round to nearest even,
//! NaN payload kept), so a checkpoint narrowed here equals one narrowed by the sdxl job it
//! replaces. A finite value that becomes infinite is not a rounding: it refuses.

use std::io::Read;

use crate::dtype::Dtype;

/// numpy `npy_floatbits_to_halfbits` with round-ties-to-even.
pub fn f32_to_f16(f: u32) -> u16 {
    let sign = ((f & 0x8000_0000) >> 16) as u16;
    let exp = f & 0x7f80_0000;
    if exp >= 0x4780_0000 {
        let sig = f & 0x007f_ffff;
        if exp == 0x7f80_0000 && sig != 0 {
            let nan = 0x7c00 + (sig >> 13) as u16;
            return sign + if nan == 0x7c00 { nan + 1 } else { nan };
        }
        return sign + 0x7c00;
    }
    if exp <= 0x3800_0000 {
        if exp < 0x3300_0000 {
            return sign;
        }
        let mut sig = (0x0080_0000 + (f & 0x007f_ffff)) >> (113 - (exp >> 23));
        if (sig & 0x3fff) != 0x1000 || (f & 0x7ff) != 0 {
            sig += 0x1000;
        }
        return sign + (sig >> 13) as u16;
    }
    let mut sig = f & 0x007f_ffff;
    if (sig & 0x3fff) != 0x1000 {
        sig += 0x1000;
    }
    sign + ((sig >> 13) as u16 + ((exp - 0x3800_0000) >> 13) as u16)
}

/// Whether this border narrows `from` to `to`: bf16/f32 to f16 only.
pub fn narrows(from: Dtype, to: Dtype) -> bool {
    to == Dtype::F16 && matches!(from, Dtype::F32 | Dtype::Bf16)
}

/// Streams `inner`'s `from` elements out as little-endian f16.
pub struct Narrow<R> {
    inner: R,
    width: usize,
    wide: bool,
    source: Vec<u8>,
    held: usize,
    out: Vec<u8>,
    at: usize,
    /// The first element (index) whose finite value f16 cannot hold.
    pub overflow: Option<u64>,
    seen: u64,
}

impl<R: Read> Narrow<R> {
    /// `None` unless `from` is a dtype this border narrows to f16.
    pub fn new(inner: R, from: Dtype) -> Option<Self> {
        let (width, wide) = match from {
            Dtype::F32 => (4, true),
            Dtype::Bf16 => (2, false),
            _ => return None,
        };
        Some(Narrow {
            inner,
            width,
            wide,
            source: vec![0; 1 << 20],
            held: 0,
            out: Vec::with_capacity(1 << 19),
            at: 0,
            overflow: None,
            seen: 0,
        })
    }

    fn refill(&mut self) -> std::io::Result<bool> {
        loop {
            let n = self.inner.read(&mut self.source[self.held..])?;
            self.held += n;
            let whole = self.held - self.held % self.width;
            if whole > 0 || n == 0 {
                self.out.clear();
                self.at = 0;
                for element in self.source[..whole].chunks_exact(self.width) {
                    let bits = if self.wide {
                        u32::from_le_bytes([element[0], element[1], element[2], element[3]])
                    } else {
                        u32::from(u16::from_le_bytes([element[0], element[1]])) << 16
                    };
                    let half = f32_to_f16(bits);
                    if half & 0x7fff == 0x7c00 && bits & 0x7f80_0000 != 0x7f80_0000 {
                        self.overflow.get_or_insert(self.seen);
                    }
                    self.seen += 1;
                    self.out.extend_from_slice(&half.to_le_bytes());
                }
                self.source.copy_within(whole..self.held, 0);
                self.held -= whole;
                if self.overflow.is_some() {
                    return Err(std::io::Error::other("value outside the f16 range"));
                }
                return Ok(whole > 0);
            }
        }
    }
}

impl<R: Read> Read for Narrow<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.at == self.out.len() && !self.refill()? {
            return Ok(0);
        }
        let n = buf.len().min(self.out.len() - self.at);
        buf[..n].copy_from_slice(&self.out[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (f32 bits, numpy 2.5.2 `astype('<f2')` bits): specials, both rounding directions,
    /// ties, subnormals, the overflow edge and NaN payloads.
    const NUMPY: [(u32, u16); 16] = [
        (0x7f80_0001, 0x7c01),
        (0x7fc0_0000, 0x7e00),
        (0xffc0_0001, 0xfe00),
        (0x7f80_2001, 0x7c01),
        (0x7fa0_0000, 0x7d00),
        (0x477f_efff, 0x7bff),
        (0x477f_f000, 0x7c00),
        (0x3300_0001, 0x0001),
        (0x3300_0000, 0x0000),
        (0x387f_e000, 0x0400),
        (0x3800_0000, 0x0200),
        (0x8000_0000, 0x8000),
        (0x3f80_0000, 0x3c00),
        (0x3f80_1000, 0x3c00),
        (0x3f80_3000, 0x3c02),
        (0xc000_0000, 0xc000),
    ];

    #[test]
    fn narrowing_is_numpys_half_conversion() {
        for (input, want) in NUMPY {
            assert_eq!(f32_to_f16(input), want, "{input:#010x}");
        }
    }

    #[test]
    fn a_finite_value_f16_cannot_hold_refuses_and_names_its_element() {
        let values: Vec<u8> = [1.0f32, 65504.0, 70000.0, f32::INFINITY]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut narrow = Narrow::new(values.as_slice(), Dtype::F32).unwrap();
        let mut out = Vec::new();
        assert!(narrow.read_to_end(&mut out).is_err());
        assert_eq!(narrow.overflow, Some(2));
        let inf: Vec<u8> = [f32::INFINITY, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = Vec::new();
        Narrow::new(inf.as_slice(), Dtype::F32)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, [0x00, 0x7c, 0x00, 0xc0]);
        let bf16 = [0x80u8, 0x3f, 0xc1, 0x7f]; // 1.0, NaN payload 0x41
        let mut out = Vec::new();
        Narrow::new(&bf16[..], Dtype::Bf16)
            .unwrap()
            .read_to_end(&mut out)
            .unwrap();
        assert_eq!(out, [0x00, 0x3c, 0x08, 0x7e]);
    }
}
