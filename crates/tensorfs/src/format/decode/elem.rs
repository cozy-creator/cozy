//! Element formats (spec §5): each code's exact f32 value.

use crate::format::Dtype;

/// 2^n for n in -149..=127, exactly.
pub fn pow2(n: i32) -> f32 {
    if n >= -126 {
        f32::from_bits(((n + 127) as u32) << 23)
    } else {
        f32::from_bits(1 << (n + 149))
    }
}

fn signed(negative: bool, v: f32) -> f32 {
    if negative {
        -v
    } else {
        v
    }
}

pub fn e4m3(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = match (e, m) {
        (15, 7) => f32::NAN,
        (0, _) => m as f32 * pow2(-9),
        _ => (8 + m) as f32 * pow2(e as i32 - 10),
    };
    signed(b & 0x80 != 0, v)
}

pub fn e5m2(b: u8) -> f32 {
    let (e, m) = ((b >> 2) & 31, b & 3);
    let v = match (e, m) {
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        (0, _) => m as f32 * pow2(-16),
        _ => (4 + m) as f32 * pow2(e as i32 - 17),
    };
    signed(b & 0x80 != 0, v)
}

/// E2M1, the low 4 bits of `n`.
pub fn e2m1(n: u8) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    signed(n & 8 != 0, MAGNITUDE[(n & 7) as usize])
}

pub fn e8m0(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        _ => pow2(b as i32 - 127),
    }
}

/// UE4M3 as ggml reads it: E4M3's exponent and mantissa, no sign and no NaN; bit 7 is ignored
/// and `0x7F` is zero.
pub fn ue4m3(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    match (b, e) {
        (0x7f, _) => 0.0,
        (_, 0) => m as f32 * pow2(-9),
        _ => (8 + m) as f32 * pow2(e as i32 - 10),
    }
}

pub fn f16(h: u16) -> f32 {
    let (e, m) = (((h >> 10) & 31) as i32, (h & 0x3ff) as u32);
    let v = match (e, m) {
        (0, _) => m as f32 * pow2(-24),
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        _ => (1024 + m) as f32 * pow2(e - 25),
    };
    signed(h & 0x8000 != 0, v)
}

pub fn bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// The value of every code of a dtype of at most 8 bits, indexed by code: `F4` (16 codes),
/// `F8_E4M3`, `F8_E5M2`, `F8_E8M0`, `I8`, `U8` (256). The `table` core reads these.
pub fn table(dtype: &Dtype) -> Option<Vec<f32>> {
    let f: fn(u8) -> f32 = match dtype {
        Dtype::F4 => return Some((0..16).map(e2m1).collect()),
        Dtype::F8_E4M3 => e4m3,
        Dtype::F8_E5M2 => e5m2,
        Dtype::F8_E8M0 => e8m0,
        Dtype::I8 => |b| b as i8 as f32,
        Dtype::U8 => |b| b as f32,
        _ => return None,
    };
    Some((0..=255).map(f).collect())
}

/// A part's elements as f32: `F32`, `BF16`, `F16` and the dtypes `table` lists (`F4` per §5,
/// element 2k in the low nibble).
pub fn floats(dtype: &Dtype, b: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F32 => (b.as_chunks::<4>().0.iter())
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::BF16 => (b.as_chunks::<2>().0.iter())
            .map(|c| bf16(u16::from_le_bytes(*c)))
            .collect(),
        Dtype::F16 => (b.as_chunks::<2>().0.iter())
            .map(|c| f16(u16::from_le_bytes(*c)))
            .collect(),
        Dtype::F4 => b
            .iter()
            .flat_map(|v| [e2m1(v & 15), e2m1(v >> 4)])
            .collect(),
        other => {
            let t =
                table(other).unwrap_or_else(|| unreachable!("no scheme reads {other} elements"));
            b.iter().map(|v| t[*v as usize]).collect()
        }
    }
}
