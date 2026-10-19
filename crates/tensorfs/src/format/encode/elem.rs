//! f32 to element codes: round to nearest, ties to even, saturating at the largest finite value
//! (a producer that rounds otherwise keeps its own). `F16`/`BF16` are the decode's `round_L`.

pub use crate::format::decode::{bf16, f16};

/// A sign-magnitude minifloat: `e` exponent bits (bias `bias`), `m` mantissa bits, subnormals
/// at exponent 0, largest finite code `max`.
struct Minifloat {
    e: u32,
    m: u32,
    bias: i32,
    max: u8,
    nan: Option<u8>,
}

impl Minifloat {
    fn value(&self, code: u8) -> f32 {
        let (e, m) = (
            (code >> self.m) as i32 & ((1 << self.e) - 1),
            (code as u32) & ((1 << self.m) - 1),
        );
        let lead = if e == 0 { 0 } else { 1 << self.m };
        (lead + m) as f32 * crate::format::decode::elem::pow2(e.max(1) - self.bias - self.m as i32)
    }

    fn encode(&self, x: f32) -> u8 {
        let sign = if x.is_sign_negative() {
            1u8 << (self.e + self.m)
        } else {
            0
        };
        if x.is_nan() {
            return self.nan.unwrap_or(0) | sign;
        }
        let a = x.abs();
        if a >= self.value(self.max) {
            return self.max | sign;
        }
        // The quantum of a's binade, never below the subnormals'.
        let exp = (a.to_bits() >> 23) as i32 - 127;
        let e = exp.max(1 - self.bias);
        let q = crate::format::decode::elem::pow2(e - self.m as i32);
        let n = (a / q).round_ties_even() as u32;
        // n·q, as (biased exponent, mantissa); n = 2^(m+1) carries into the next binade.
        let (mut be, mut man) = ((e + self.bias) as u32, n);
        if man >= 2 << self.m {
            be += 1;
            man >>= 1;
        }
        let code = if man < 1 << self.m {
            man
        } else {
            (be << self.m) | (man - (1 << self.m))
        };
        code as u8 | sign
    }
}

const E4M3: Minifloat = Minifloat {
    e: 4,
    m: 3,
    bias: 7,
    max: 0x7e,
    nan: Some(0x7f),
};
const E5M2: Minifloat = Minifloat {
    e: 5,
    m: 2,
    bias: 15,
    max: 0x7b,
    nan: Some(0x7f),
};
const E2M1: Minifloat = Minifloat {
    e: 2,
    m: 1,
    bias: 1,
    max: 0x7,
    nan: None,
};

pub fn e4m3(x: f32) -> u8 {
    E4M3.encode(x)
}

pub fn e5m2(x: f32) -> u8 {
    E5M2.encode(x)
}

/// The E2M1 nibble; NaN has no code and encodes as zero.
pub fn e2m1(x: f32) -> u8 {
    E2M1.encode(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::decode::elem;

    /// Every finite code round-trips; each midpoint between neighbours goes to the even code;
    /// past the largest finite value saturates.
    fn exhaustive(sign: u8, decode: fn(u8) -> f32, encode: fn(f32) -> u8, max: u8) {
        let mut finite: Vec<(f32, u8)> = (0..sign)
            .map(|c| (decode(c), c))
            .filter(|(v, _)| v.is_finite())
            .collect();
        finite.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (v, c) in &finite {
            assert_eq!(encode(*v), *c, "{v}");
            assert_eq!(encode(-*v), *c | sign, "-{v}");
        }
        for w in finite.windows(2) {
            let ((lo, a), (hi, b)) = (w[0], w[1]);
            let mid = (lo + hi) / 2.0;
            let even = if a % 2 == 0 { a } else { b };
            assert_eq!(encode(mid), even, "{lo}..{hi}");
            assert_eq!(encode(mid.next_up()), b);
            assert_eq!(encode(mid.next_down()), a);
        }
        assert_eq!(encode(decode(max) * 4.0), max);
        assert_eq!(encode(f32::INFINITY), max);
        assert_eq!(encode(-f32::INFINITY), max | sign);
    }

    #[test]
    fn element_codes_round_trip_and_tie_to_even() {
        exhaustive(0x80, elem::e4m3, e4m3, 0x7e);
        exhaustive(0x80, elem::e5m2, e5m2, 0x7b);
        exhaustive(0x8, elem::e2m1, e2m1, 0x7);
        assert_eq!(e4m3(f32::NAN), 0x7f);
        assert_eq!(
            e4m3(elem::pow2(-10)),
            0,
            "half the smallest subnormal ties to zero"
        );
        assert_eq!(e4m3(elem::pow2(-10).next_up()), 1);
    }
}
