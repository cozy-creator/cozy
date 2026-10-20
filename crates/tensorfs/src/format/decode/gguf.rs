//! `gguf.<type>` (spec §6 `gguf.*`): ggml's block structs as llama.cpp writes them in a GGUF, each
//! element in ggml's own f32 order (`dequantize_row_*`, `ggml-quants.c` at llama.cpp ff5a233).

use super::blocks::{f16_at, u32_at};
use super::plan::{Core, Elems, Post, Struct};
use super::{elem, one, Allowed, Ctx, Dtypes, Param, Part, Plan, Scheme, Shape, View};
use crate::format::{Encoding, Result};

/// A ggml type: id, name, elements per block, bytes per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Type {
    pub id: u32,
    pub name: &'static str,
    pub block: u64,
    pub size: u64,
}

const fn t(id: u32, name: &'static str, block: u64, size: u64) -> Type {
    Type {
        id,
        name,
        block,
        size,
    }
}

/// ggml's types (`ggml.h`, `ggml.c` type traits) at llama.cpp ff5a233.
pub const TYPES: [Type; 35] = [
    t(0, "f32", 1, 4),
    t(1, "f16", 1, 2),
    t(2, "q4_0", 32, 18),
    t(3, "q4_1", 32, 20),
    t(6, "q5_0", 32, 22),
    t(7, "q5_1", 32, 24),
    t(8, "q8_0", 32, 34),
    t(9, "q8_1", 32, 36),
    t(10, "q2_k", 256, 84),
    t(11, "q3_k", 256, 110),
    t(12, "q4_k", 256, 144),
    t(13, "q5_k", 256, 176),
    t(14, "q6_k", 256, 210),
    t(15, "q8_k", 256, 292),
    t(16, "iq2_xxs", 256, 66),
    t(17, "iq2_xs", 256, 74),
    t(18, "iq3_xxs", 256, 98),
    t(19, "iq1_s", 256, 50),
    t(20, "iq4_nl", 32, 18),
    t(21, "iq3_s", 256, 110),
    t(22, "iq2_s", 256, 82),
    t(23, "iq4_xs", 256, 136),
    t(24, "i8", 1, 1),
    t(25, "i16", 1, 2),
    t(26, "i32", 1, 4),
    t(27, "i64", 1, 8),
    t(28, "f64", 1, 8),
    t(29, "iq1_m", 256, 56),
    t(30, "bf16", 1, 2),
    t(34, "tq1_0", 256, 54),
    t(35, "tq2_0", 256, 66),
    t(39, "mxfp4", 32, 17),
    t(40, "nvfp4", 64, 36),
    t(41, "q1_0", 128, 18),
    t(42, "q2_0", 64, 18),
];

impl Type {
    pub fn by_id(id: u32) -> Option<Type> {
        TYPES.into_iter().find(|t| t.id == id)
    }

    pub fn by_name(name: &str) -> Option<Type> {
        TYPES.into_iter().find(|t| t.name == name)
    }

    /// `numel` elements' bytes, if they are whole blocks.
    pub fn bytes(self, numel: u64) -> Option<u64> {
        numel
            .is_multiple_of(self.block)
            .then(|| (numel / self.block).checked_mul(self.size))?
    }

    /// `gguf.<type> {block_size}`.
    pub fn encoding(self) -> Encoding {
        super::super::encode::encoding(
            &format!("gguf.{}", self.name),
            &[("block_size", self.block.into())],
        )
    }
}

/// The `blocks` part's shape for logical `shape`: `[lead…, C/n, S]`, or `[N/n, S]` when the last
/// extent is not whole blocks. Structs follow the row-major element order either way.
pub fn blocks_shape(ty: Type, shape: &[u64]) -> Option<Vec<u64>> {
    let numel = shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b))?;
    if !numel.is_multiple_of(ty.block) {
        return None;
    }
    Some(match shape.split_last() {
        Some((c, lead)) if c.is_multiple_of(ty.block) => [lead, &[c / ty.block, ty.size]].concat(),
        _ => vec![numel / ty.block, ty.size],
    })
}

/// ggml's `kvalues_fp4`: E2M1 doubled (spec Appendix A.2).
pub const K: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

const BLOCKS: Part = Part::req("blocks", Dtypes::Of(&["U8"]), shape);

macro_rules! scheme {
    ($name:literal, $n:literal) => {
        scheme!($name, $n, &[BLOCKS])
    };
    ($name:literal, $n:literal, $parts:expr) => {
        Scheme {
            name: concat!("gguf.", $name),
            view: View::Flat,
            params: &[Param {
                key: "block_size",
                allowed: Allowed::Uint(&[$n]),
            }],
            parts: $parts,
            decode,
        }
    };
}

pub const Q4_0: Scheme = scheme!("q4_0", 32);
pub const Q4_1: Scheme = scheme!("q4_1", 32);
pub const Q5_0: Scheme = scheme!("q5_0", 32);
pub const Q5_1: Scheme = scheme!("q5_1", 32);
pub const Q8_0: Scheme = scheme!("q8_0", 32);
pub const Q2_K: Scheme = scheme!("q2_k", 256);
pub const Q3_K: Scheme = scheme!("q3_k", 256);
pub const Q4_K: Scheme = scheme!("q4_k", 256);
pub const Q5_K: Scheme = scheme!("q5_k", 256);
pub const Q6_K: Scheme = scheme!("q6_k", 256);
pub const MXFP4: Scheme = scheme!("mxfp4", 32);
pub const NVFP4: Scheme = scheme!(
    "nvfp4",
    64,
    &[
        BLOCKS,
        Part::opt("scale", Dtypes::Of(&["F32"]), one),
        Part::opt("input_scale", Dtypes::Of(&["F32"]), one),
    ]
);
pub const Q1_0: Scheme = scheme!("q1_0", 128);
pub const Q2_0: Scheme = scheme!("q2_0", 64);

fn ty(c: &Ctx) -> Type {
    Type::by_name(&c.scheme["gguf.".len()..]).expect("every row names a ggml type")
}

fn shape(c: &Ctx) -> Result<Shape> {
    let ty = ty(c);
    match blocks_shape(ty, c.shape) {
        Some(s) => Ok(Shape::Exact(s)),
        None => Err(c.fail(format!("{:?} is not whole blocks of {}", c.shape, ty.block))),
    }
}

/// The struct types a scheme decodes, in the GPU kernels' order (`tensorfs/kernels.py` `STRUCTS`).
pub static STRUCTS: [Struct; 14] = [
    st("q4_0", 32, 18, q4_0),
    st("q4_1", 32, 20, q4_1),
    st("q5_0", 32, 22, q5_0),
    st("q5_1", 32, 24, q5_1),
    st("q8_0", 32, 34, q8_0),
    st("q2_k", 256, 84, q2_k),
    st("q3_k", 256, 110, q3_k),
    st("q4_k", 256, 144, q4_k),
    st("q5_k", 256, 176, q5_k),
    st("q6_k", 256, 210, q6_k),
    st("mxfp4", 32, 17, mxfp4),
    st("nvfp4", 64, 36, nvfp4),
    st("q1_0", 128, 18, q1_0),
    st("q2_0", 64, 18, q2_0),
];

const fn st(name: &'static str, n: usize, size: usize, decode: fn(&[u8], &mut [f32])) -> Struct {
    Struct {
        name,
        n,
        size,
        decode,
    }
}

fn decode<'a>(c: &Ctx<'a>) -> Result<Plan<'a>> {
    let name = ty(c).name;
    let kind = STRUCTS
        .iter()
        .find(|s| s.name == name)
        .expect("every row's type is a struct");
    let post = match c.has("scale") {
        true => Post::One(Elems::part(c.part("scale"))),
        false => Post::None,
    };
    Ok(Plan {
        core: Core::Structs {
            bytes: c.part("blocks").bytes,
            kind,
        },
        post,
    })
}

/// `q4_0`'s nibble order: element `j` in `qs[j]`'s low nibble, `j + 16` in its high one.
fn nibbles(qs: &[u8], y: &mut [f32], v: impl Fn(u8) -> f32) {
    let half = qs.len();
    for (j, q) in qs.iter().enumerate() {
        y[j] = v(q & 15);
        y[j + half] = v(q >> 4);
    }
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    nibbles(&b[2..18], y, |q| (q as i32 - 8) as f32 * d);
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    nibbles(&b[4..20], y, |q| q as f32 * d + m);
}

/// `q5_0`/`q5_1`: `q4_0`'s nibbles plus bit `j` of `qh` as the fifth.
fn fifth(qh: u32, qs: &[u8], y: &mut [f32], v: impl Fn(u32) -> f32) {
    for (j, q) in qs.iter().enumerate() {
        y[j] = v((q & 15) as u32 | ((qh >> j) << 4 & 0x10));
        y[j + 16] = v((q >> 4) as u32 | ((qh >> (j + 12)) & 0x10));
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    fifth(u32_at(b, 2), &b[6..22], y, |q| (q as i32 - 16) as f32 * d);
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (f16_at(b, 0), f16_at(b, 2));
    fifth(u32_at(b, 4), &b[8..24], y, |q| q as f32 * d + m);
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (j, q) in b[2..34].iter().enumerate() {
        y[j] = *q as i8 as f32 * d;
    }
}

fn q2_k(b: &[u8], y: &mut [f32]) {
    let (scales, qs) = (&b[0..16], &b[16..80]);
    let (d, min) = (f16_at(b, 80), f16_at(b, 82));
    let mut t = 0;
    for h in 0..2 {
        let q = &qs[32 * h..32 * h + 32];
        for j in 0..4 {
            for s in 0..2 {
                let sc = scales[8 * h + 2 * j + s];
                let (dl, ml) = (d * (sc & 15) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    y[t] = dl * ((q[16 * s + l] >> (2 * j)) & 3) as f32 - ml;
                    t += 1;
                }
            }
        }
    }
}

/// `q3_k`'s 16 six-bit scales, packed in 12 bytes.
fn q3_scales(s: &[u8]) -> [i32; 16] {
    std::array::from_fn(|k| {
        let low = if k < 8 { s[k] & 15 } else { s[k - 8] >> 4 };
        let high = (s[8 + k % 4] >> (2 * (k / 4))) & 3;
        (low | high << 4) as i32 - 32
    })
}

fn q3_k(b: &[u8], y: &mut [f32]) {
    let (hmask, qs) = (&b[0..32], &b[32..96]);
    let scales = q3_scales(&b[96..108]);
    let d = f16_at(b, 108);
    let mut t = 0;
    for h in 0..2 {
        let q = &qs[32 * h..32 * h + 32];
        for j in 0..4 {
            let m = 1u8 << (4 * h + j);
            for s in 0..2 {
                let dl = d * scales[8 * h + 2 * j + s] as f32;
                for l in 0..16 {
                    let low = ((q[16 * s + l] >> (2 * j)) & 3) as i32;
                    let high = if hmask[16 * s + l] & m != 0 { 0 } else { 4 };
                    y[t] = dl * (low - high) as f32;
                    t += 1;
                }
            }
        }
    }
}

/// `get_scale_min_k4`: sub-block `a`'s six-bit scale and min.
fn scale_min(a: usize, s: &[u8]) -> (f32, f32) {
    let (c, e) = if a < 4 {
        (s[a] & 63, s[a + 4] & 63)
    } else {
        (
            (s[a + 4] & 15) | ((s[a - 4] >> 6) << 4),
            (s[a + 4] >> 4) | ((s[a] >> 6) << 4),
        )
    };
    (c as f32, e as f32)
}

/// `q4_k` and `q5_k`: 8 sub-blocks of 32, `(d × c) × q − dmin × e`; `qh` adds `q5_k`'s fifth bit.
fn k_quant(b: &[u8], qs: &[u8], qh: Option<&[u8]>, y: &mut [f32]) {
    let (d, min, scales) = (f16_at(b, 0), f16_at(b, 2), &b[4..16]);
    for a in 0..8 {
        let (c, e) = scale_min(a, scales);
        let (dl, ml) = (d * c, min * e);
        let q = &qs[32 * (a / 2)..32 * (a / 2) + 32];
        for l in 0..32 {
            let low = if a % 2 == 0 { q[l] & 15 } else { q[l] >> 4 };
            let high = qh.map_or(0, |h| ((h[l] >> a) & 1) << 4);
            y[32 * a + l] = dl * (low + high) as f32 - ml;
        }
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    k_quant(b, &b[16..144], None, y);
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    k_quant(b, &b[48..176], Some(&b[16..48]), y);
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let (ql, qh, d) = (&b[0..128], &b[128..192], f16_at(b, 208));
    for h in 0..2 {
        let (ql, qh, sc) = (&ql[64 * h..], &qh[32 * h..], &b[192 + 8 * h..]);
        for l in 0..32 {
            for j in 0..4 {
                let low = if j < 2 {
                    ql[l + 32 * j] & 15
                } else {
                    ql[l + 32 * (j - 2)] >> 4
                };
                let q = (low | ((qh[l] >> (2 * j)) & 3) << 4) as i32 - 32;
                y[128 * h + 32 * j + l] = d * sc[l / 16 + 2 * j] as i8 as f32 * q as f32;
            }
        }
    }
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    // 2^(e − 128), denormal for e < 2: ggml's `ggml_e8m0_to_fp32_half`.
    let d = elem::pow2(b[0] as i32 - 128);
    nibbles(&b[1..17], y, |q| K[q as usize] as f32 * d);
}

fn nvfp4(b: &[u8], y: &mut [f32]) {
    for s in 0..4 {
        let d = elem::ue4m3(b[s]) * 0.5;
        nibbles(
            &b[4 + 8 * s..12 + 8 * s],
            &mut y[16 * s..16 * s + 16],
            |q| K[q as usize] as f32 * d,
        );
    }
}

fn q1_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (j, v) in y.iter_mut().enumerate() {
        *v = if (b[2 + j / 8] >> (j % 8)) & 1 == 1 {
            d
        } else {
            -d
        };
    }
}

fn q2_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (j, v) in y.iter_mut().enumerate() {
        *v = (((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32 * d;
    }
}
