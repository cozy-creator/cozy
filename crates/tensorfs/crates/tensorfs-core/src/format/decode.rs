//! The schemes (spec §6): their complete params for writers, and the reference CPU decode. Each
//! element is computed in f32 by the scheme's formula, then rounded once to the logical dtype
//! (round to nearest even; NaN becomes the canonical quiet NaN). The Runtime's GPU decoders
//! match it bit for bit.

use std::collections::BTreeMap;

use super::{Dtype, Encoding, Error, Result, Value};

/// The schemes this reader decodes. A plain tensor needs no scheme.
pub const SCHEMES: [&str; 6] = [
    "fp8.rowwise",
    "fp8.tensorwise",
    "int8.rowwise",
    "int8.convrot",
    "mxfp8",
    "nvfp4",
];

/// One part's stored bytes, as the header declares them.
#[derive(Debug, Clone, Copy)]
pub struct PartBytes<'a> {
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub bytes: &'a [u8],
}

fn encoding(scheme: &str, params: &[(&str, Value)]) -> Encoding {
    Encoding {
        scheme: scheme.into(),
        params: params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
    }
}

/// Each scheme with its complete params: a reader needs nothing but these and the spec.
impl Encoding {
    /// An fp8 weight `[R, C]`, one F32 scale per row.
    pub fn fp8_rowwise(weight: &Dtype) -> Encoding {
        let p = [("scale", "per_row".into()), ("scale_dtype", "F32".into())];
        encoding("fp8.rowwise", &[&p[..], &[("weight_dtype", weight.name().into())]].concat())
    }
    /// An fp8 weight of any shape, one F32 scale.
    pub fn fp8_tensorwise(weight: &Dtype) -> Encoding {
        let p = [("scale", "per_tensor".into()), ("scale_dtype", "F32".into())];
        encoding("fp8.tensorwise", &[&p[..], &[("weight_dtype", weight.name().into())]].concat())
    }
    /// An I8 weight `[R, C]`, one F32 scale per row.
    pub fn int8_rowwise() -> Encoding {
        encoding(
            "int8.rowwise",
            &[
                ("scale", "per_row".into()),
                ("scale_dtype", "F32".into()),
                ("weight_dtype", "I8".into()),
            ],
        )
    }
    /// An I8 weight `[R, C]` Hadamard-rotated in groups of `block` columns, one F32 scale per row.
    pub fn int8_convrot(block: u64) -> Encoding {
        encoding(
            "int8.convrot",
            &[
                ("block", block.into()),
                ("rotation", "hadamard".into()),
                ("scale", "per_row".into()),
                ("scale_dtype", "F32".into()),
                ("weight_dtype", "I8".into()),
            ],
        )
    }
    /// OCP MX: an fp8 weight `[R, C]`, one E8M0 scale per row and block of 32 columns.
    pub fn mxfp8(weight: &Dtype) -> Encoding {
        encoding(
            "mxfp8",
            &[
                ("block", 32u64.into()),
                ("scale", "per_block".into()),
                ("scale_dtype", "F8_E8M0".into()),
                ("weight_dtype", weight.name().into()),
            ],
        )
    }
    /// NVIDIA NVFP4: an F4 weight `[R, C]`, one E4M3 scale per row and block of 16 columns, and
    /// one F32 global scale.
    pub fn nvfp4() -> Encoding {
        encoding(
            "nvfp4",
            &[
                ("block", 16u64.into()),
                ("global_scale", "per_tensor".into()),
                ("global_scale_dtype", "F32".into()),
                ("scale", "per_block".into()),
                ("scale_dtype", "F8_E4M3".into()),
                ("weight_dtype", "F4".into()),
            ],
        )
    }
}

/// The logical tensor's bytes from its encoded parts.
pub fn decode(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
) -> Result<Vec<u8>> {
    use Dtype::*;
    let scheme = encoding.scheme.as_str();
    let (required, optional): (&[&str], &[&str]) = match scheme {
        "fp8.rowwise" | "int8.rowwise" | "int8.convrot" | "mxfp8" => (&["scale", "weight"], &[]),
        "fp8.tensorwise" => (&["scale", "weight"], &["input_scale"]),
        "nvfp4" => (&["global_scale", "scale", "weight"], &["input_scale"]),
        other => return Err(Error::UnknownScheme(other.into())),
    };
    let fail = |detail: String| Error::Decode(format!("{scheme}: {detail}"));
    for role in parts.keys() {
        if !required.contains(role) && !optional.contains(role) {
            return Err(fail(format!("unknown part {role:?}")));
        }
    }
    if let Some(role) = required.iter().find(|r| !parts.contains_key(*r)) {
        return Err(fail(format!("missing part {role:?}")));
    }
    // Params a scheme defines must be present and right; others are ignored.
    let param = |key: &str| encoding.params.get(key);
    let text = |key: &str, allowed: &[&str]| match param(key).and_then(Value::as_str) {
        Some(v) if allowed.contains(&v) => Ok(v),
        _ => Err(fail(format!("param {key} is not one of {allowed:?}"))),
    };
    let fp8 = ["F8_E4M3", "F8_E5M2"];
    let weight = match scheme {
        "fp8.rowwise" | "fp8.tensorwise" | "mxfp8" => text("weight_dtype", &fp8)?,
        "nvfp4" => text("weight_dtype", &["F4"])?,
        _ => text("weight_dtype", &["I8"])?,
    };
    let (granularity, scale_dtype) = match scheme {
        "fp8.tensorwise" => ("per_tensor", "F32"),
        "mxfp8" => ("per_block", "F8_E8M0"),
        "nvfp4" => ("per_block", "F8_E4M3"),
        _ => ("per_row", "F32"),
    };
    text("scale", &[granularity])?;
    text("scale_dtype", &[scale_dtype])?;
    if !matches!(dtype, F32 | BF16 | F16) {
        return Err(fail(format!(
            "logical dtype {dtype} is not F32, BF16 or F16"
        )));
    }
    let numel = shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
    let (rows, cols) = match (shape, numel) {
        ([r, c], Some(_)) => (*r as usize, *c as usize),
        (_, Some(n)) if scheme == "fp8.tensorwise" => (1, n as usize),
        _ => return Err(fail(format!("logical shape {shape:?} is not rank 2"))),
    };
    let part = |role: &str, dtype: &str, want: Option<&[u64]>, count: u64| {
        let p = parts[role];
        if p.dtype.name() != dtype {
            return Err(fail(format!("part {role} is {}, not {dtype}", p.dtype)));
        }
        let ok = p.dtype.byte_len(p.shape) == Some(p.bytes.len() as u64)
            && match want {
                Some(s) => p.shape == s,
                None => p.shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b)) == Some(count),
            };
        if !ok {
            return Err(fail(format!("part {role} has shape {:?}", p.shape)));
        }
        Ok(p)
    };
    // `whole`: the columns are whole blocks.
    let block = |allowed: &[u64], whole: bool| match param("block").and_then(Value::as_u64) {
        Some(b) if allowed.contains(&b) && (!whole || cols % b as usize == 0) => Ok(b as usize),
        _ => Err(fail(format!(
            "param block is not one of {allowed:?}, or {cols} columns are not whole blocks"
        ))),
    };
    let w = part("weight", weight, Some(shape), 0)?;
    let out: Vec<f32> = match scheme {
        "fp8.rowwise" | "int8.rowwise" => {
            let s = floats(part("scale", "F32", None, rows as u64)?);
            let w = floats(w);
            w.iter().enumerate().map(|(i, v)| v * s[i / cols]).collect()
        }
        "fp8.tensorwise" => {
            let s = narrow(dtype, floats(part("scale", "F32", None, 1)?)[0]);
            if parts.contains_key("input_scale") {
                part("input_scale", "F32", None, 1)?;
            }
            floats(w).iter().map(|v| v * s).collect()
        }
        "int8.convrot" => {
            let block = block(&[4, 16, 64, 256, 1024, 4096, 16384, 65536], true)?;
            text("rotation", &["hadamard"])?;
            let k = block.trailing_zeros() / 2;
            let s = floats(part("scale", "F32", None, rows as u64)?);
            let unit = pow2(-(k as i32));
            let mut out = Vec::with_capacity(rows * cols);
            for (g, group) in w.bytes.chunks(block).enumerate() {
                let row = g * block / cols;
                let mut h: Vec<i32> = group.iter().map(|b| *b as i8 as i32).collect();
                hadamard(&mut h);
                out.extend(h.iter().map(|v| *v as f32 * s[row] * unit));
            }
            out
        }
        "mxfp8" => {
            block(&[32], false)?;
            let blocks = cols.div_ceil(32);
            let s = floats(part("scale", "F8_E8M0", Some(&[rows as u64, blocks as u64]), 0)?);
            let w = floats(w);
            w.iter()
                .enumerate()
                .map(|(i, v)| v * s[i / cols * blocks + i % cols / 32])
                .collect()
        }
        "nvfp4" => {
            block(&[16], true)?;
            text("global_scale", &["per_tensor"])?;
            text("global_scale_dtype", &["F32"])?;
            let shape = [rows as u64, cols as u64 / 16];
            let s = floats(part("scale", "F8_E4M3", Some(&shape), 0)?);
            let g = floats(part("global_scale", "F32", None, 1)?)[0];
            if parts.contains_key("input_scale") {
                part("input_scale", "F32", None, 1)?;
            }
            let w = floats(w);
            w.iter()
                .enumerate()
                .map(|(i, v)| v * s[i / 16] * g)
                .collect()
        }
        _ => unreachable!("matched above"),
    };
    Ok(round(dtype, &out))
}

/// Element values of a part, as f32.
fn floats(p: PartBytes) -> Vec<f32> {
    let b = p.bytes;
    match p.dtype {
        Dtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        Dtype::I8 => b.iter().map(|v| *v as i8 as f32).collect(),
        Dtype::F8_E4M3 => b.iter().map(|v| e4m3(*v)).collect(),
        Dtype::F8_E8M0 => (b.iter())
            .map(|v| match v {
                255 => f32::NAN,
                _ => pow2(*v as i32 - 127),
            })
            .collect(),
        Dtype::F8_E5M2 => b.iter().map(|v| e5m2(*v)).collect(),
        Dtype::F4 => b
            .iter()
            .flat_map(|v| [e2m1(v & 15), e2m1(v >> 4)])
            .collect(),
        other => unreachable!("no scheme reads {other} elements"),
    }
}

/// 2^n for n in -149..=127, exactly.
fn pow2(n: i32) -> f32 {
    if n >= -126 {
        f32::from_bits(((n + 127) as u32) << 23)
    } else {
        f32::from_bits(1 << (n + 149))
    }
}

fn signed(byte: u8, sign_bit: u8, v: f32) -> f32 {
    if byte & sign_bit != 0 {
        -v
    } else {
        v
    }
}

fn e4m3(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    let v = match (e, m) {
        (15, 7) => f32::NAN,
        (0, _) => m as f32 * pow2(-9),
        _ => (8 + m) as f32 * pow2(e as i32 - 10),
    };
    signed(b, 0x80, v)
}

fn e5m2(b: u8) -> f32 {
    let (e, m) = ((b >> 2) & 31, b & 3);
    let v = match (e, m) {
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        (0, _) => m as f32 * pow2(-16),
        _ => (4 + m) as f32 * pow2(e as i32 - 17),
    };
    signed(b, 0x80, v)
}

fn e2m1(n: u8) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    signed(n, 8, MAGNITUDE[(n & 7) as usize])
}

/// In place, `x · H` for H = H4 ⊗ … ⊗ H4 (unnormalized, entries ±1), x.len() a power of 4.
fn hadamard(x: &mut [i32]) {
    let mut stride = 1;
    while stride < x.len() {
        for base in (0..x.len()).filter(|i| (i / stride) % 4 == 0) {
            let [a, b, c, d] = [0, 1, 2, 3].map(|j| x[base + j * stride]);
            let out = [a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a];
            for (j, v) in out.into_iter().enumerate() {
                x[base + j * stride] = v;
            }
        }
        stride *= 4;
    }
}

/// `x` rounded to the logical dtype and read back as f32.
fn narrow(dtype: &Dtype, x: f32) -> f32 {
    match dtype {
        Dtype::BF16 => f32::from_bits((bf16(x) as u32) << 16),
        Dtype::F16 => f16_to_f32(f16(x)),
        _ => x,
    }
}

/// Little-endian bytes of `xs` in `dtype` (F32, BF16 or F16).
fn round(dtype: &Dtype, xs: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F32 => xs
            .iter()
            .flat_map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() }.to_le_bytes())
            .collect(),
        Dtype::BF16 => xs.iter().flat_map(|x| bf16(*x).to_le_bytes()).collect(),
        _ => xs.iter().flat_map(|x| f16(*x).to_le_bytes()).collect(),
    }
}

fn bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let (exp, man) = (((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
    let e = exp - 112;
    let (r, rem, half) = if exp == 0xff || e >= 31 {
        return sign | 0x7c00;
    } else if e <= 0 {
        if e < -10 {
            return sign;
        }
        let shift = (14 - e) as u32;
        let m = man | 0x80_0000;
        (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1))
    } else {
        (((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000)
    };
    sign | (r + (rem > half || (rem == half && r & 1 == 1)) as u32) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let (e, m) = (((h >> 10) & 31) as i32, (h & 0x3ff) as u32);
    let v = match e {
        0 => m as f32 * pow2(-24),
        _ => (1024 + m) as f32 * pow2(e - 25),
    };
    signed((h >> 8) as u8, 0x80, v)
}
