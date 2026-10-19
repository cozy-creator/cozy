//! The schemes (spec §6): each one's complete encoding for writers, and the reference CPU decode.
//! Each element is computed in f32 by the scheme's formula, then rounded once to the logical
//! dtype (round to nearest even; NaN becomes the canonical quiet NaN). The Runtime's GPU decoders
//! match it bit for bit.

use std::collections::BTreeMap;

use super::{Dtype, Encoding, Error, Result, Value};

/// The schemes this reader decodes. A plain tensor needs no scheme.
pub const SCHEMES: [&str; 7] = [
    "fp8.tensorwise",
    "fp8.rowwise",
    "int8.tensorwise",
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

/// How block scales are stored: `[rows, blocks]` row-major, or cuBLAS's 128×4 tiles (ComfyUI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLayout {
    RowMajor,
    Cublas128x4,
}

/// Which nibble of a byte holds an F4 pair's even element: ModelOpt's low, ComfyUI's high.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder {
    LowFirst,
    HighFirst,
}

/// How nvfp4's per-tensor scales, `scale_2` and `input_scale`, apply: multiplied (ModelOpt,
/// ComfyUI) or divided, being reciprocals (compressed-tensors' `weight_global_scale` and
/// `input_global_scale`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalScale {
    Multiply,
    Divide,
}

impl ScaleLayout {
    pub fn name(self) -> &'static str {
        match self {
            ScaleLayout::RowMajor => "row_major",
            ScaleLayout::Cublas128x4 => "cublas_128x4",
        }
    }
}

impl NibbleOrder {
    pub fn name(self) -> &'static str {
        match self {
            NibbleOrder::LowFirst => "low_first",
            NibbleOrder::HighFirst => "high_first",
        }
    }
}

impl GlobalScale {
    pub fn name(self) -> &'static str {
        match self {
            GlobalScale::Multiply => "multiply",
            GlobalScale::Divide => "divide",
        }
    }
}

fn encoding(scheme: &str, params: &[(&str, Value)]) -> Encoding {
    Encoding {
        scheme: scheme.into(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

/// Each scheme with every param its decode reads; part dtypes and shapes say the rest.
impl Encoding {
    /// One scale for the tensor, stored as the producer wrote it: `F32`, `BF16` or `F16`.
    pub fn fp8_tensorwise(scale: &Dtype) -> Encoding {
        encoding("fp8.tensorwise", &[("scale_dtype", scale.name().into())])
    }
    /// `axis` 0: a scale per row; 1: per column.
    pub fn fp8_rowwise(axis: u64, scale: &Dtype) -> Encoding {
        let p = [("axis", axis.into()), ("scale_dtype", scale.name().into())];
        encoding("fp8.rowwise", &p)
    }
    pub fn int8_tensorwise(scale: &Dtype) -> Encoding {
        encoding("int8.tensorwise", &[("scale_dtype", scale.name().into())])
    }
    pub fn int8_rowwise(axis: u64, scale: &Dtype) -> Encoding {
        let p = [("axis", axis.into()), ("scale_dtype", scale.name().into())];
        encoding("int8.rowwise", &p)
    }
    pub fn int8_convrot(rotation_size: u64, scale: &Dtype) -> Encoding {
        let p = [
            ("rotation", "hadamard".into()),
            ("rotation_size", rotation_size.into()),
            ("scale_dtype", scale.name().into()),
        ];
        encoding("int8.convrot", &p)
    }
    pub fn mxfp8(scale_layout: ScaleLayout) -> Encoding {
        let p = [
            ("block_size", 32u64.into()),
            ("scale_layout", scale_layout.name().into()),
        ];
        encoding("mxfp8", &p)
    }
    pub fn nvfp4(global: GlobalScale, nibbles: NibbleOrder, layout: ScaleLayout) -> Encoding {
        let p = [
            ("block_size", 16u64.into()),
            ("global_scale", global.name().into()),
            ("nibble_order", nibbles.name().into()),
            ("scale_layout", layout.name().into()),
        ];
        encoding("nvfp4", &p)
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
    let (required, optional, defined): (&[&str], &[&str], &[&str]) = match scheme {
        "fp8.tensorwise" | "int8.tensorwise" => {
            (&["scale", "weight"], &["input_scale"], &["scale_dtype"])
        }
        "fp8.rowwise" | "int8.rowwise" => (&["scale", "weight"], &[], &["axis", "scale_dtype"]),
        "int8.convrot" => (
            &["scale", "weight"],
            &[],
            &["rotation", "rotation_size", "scale_dtype"],
        ),
        "mxfp8" => (&["scale", "weight"], &[], &["block_size", "scale_layout"]),
        "nvfp4" => (
            &["scale", "scale_2", "weight"],
            &["input_scale"],
            &["block_size", "global_scale", "nibble_order", "scale_layout"],
        ),
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
    // Params drive the decode: one this reader does not know is a refusal, not a default.
    if let Some(key) = encoding
        .params
        .keys()
        .find(|k| !defined.contains(&k.as_str()))
    {
        return Err(fail(format!("unknown param {key:?}")));
    }
    let text = |key: &str, allowed: &[&str]| match encoding.params.get(key).and_then(Value::as_str)
    {
        Some(v) if allowed.contains(&v) => Ok(v),
        _ => Err(fail(format!("param {key} is not one of {allowed:?}"))),
    };
    let uint = |key: &str, allowed: &[u64]| match encoding.params.get(key).and_then(Value::as_u64) {
        Some(v) if allowed.contains(&v) => Ok(v as usize),
        _ => Err(fail(format!("param {key} is not one of {allowed:?}"))),
    };
    // A free-dtype scale, as the producer stored it; read exactly, never rounded first.
    let scale_dtype = || match text("scale_dtype", &["F32", "BF16", "F16"])? {
        "F32" => Ok(F32),
        "BF16" => Ok(BF16),
        _ => Ok(F16),
    };
    if !matches!(dtype, F32 | BF16 | F16) {
        return Err(fail(format!(
            "logical dtype {dtype} is not F32, BF16 or F16"
        )));
    }
    // Every scheme but the tensorwise two views the tensor as [rows, cols]: the last axis, and
    // every leading axis as rows.
    let numel = shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
    let (rows, cols) = match (shape, numel) {
        (_, Some(n)) if scheme.ends_with(".tensorwise") => (1, n as usize),
        ([.., c], Some(n)) if shape.len() >= 2 && *c > 0 => ((n / c) as usize, *c as usize),
        ([.., _], Some(0)) if shape.len() >= 2 => (0, 1),
        _ => {
            return Err(fail(format!(
                "logical shape {shape:?} is not rank 2 or more"
            )));
        }
    };
    let part = |role: &str, dtypes: &[Dtype], want: Option<&[u64]>, count: u64| {
        let p = parts[role];
        if !dtypes.contains(p.dtype) {
            return Err(fail(format!(
                "part {role} is {}, not one of {dtypes:?}",
                p.dtype
            )));
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
    // The stored scale shape for `blocks` per row, and whether it is cuBLAS-tiled.
    let scales = |layout: &str, blocks: usize| match layout {
        "row_major" => (
            [&shape[..shape.len() - 1], &[blocks as u64]].concat(),
            false,
        ),
        _ => (
            vec![
                rows.div_ceil(128) as u64 * 128,
                blocks.div_ceil(4) as u64 * 4,
            ],
            true,
        ),
    };
    let fp8 = [F8_E4M3, F8_E5M2];
    let out: Vec<f32> = match scheme {
        "fp8.tensorwise" | "int8.tensorwise" => {
            let w = part(
                "weight",
                if scheme == "fp8.tensorwise" {
                    &fp8
                } else {
                    &[I8]
                },
                Some(shape),
                0,
            )?;
            let s = floats(part("scale", &[scale_dtype()?], None, 1)?)[0];
            if parts.contains_key("input_scale") {
                part("input_scale", &[F32], None, 1)?;
            }
            floats(w).iter().map(|v| v * s).collect()
        }
        "fp8.rowwise" | "int8.rowwise" => {
            let w = part(
                "weight",
                if scheme == "fp8.rowwise" { &fp8 } else { &[I8] },
                Some(shape),
                0,
            )?;
            let axis = uint("axis", &[0, 1])?;
            let s = floats(part(
                "scale",
                &[scale_dtype()?],
                None,
                [rows, cols][axis] as u64,
            )?);
            let at = |i: usize| if axis == 0 { i / cols } else { i % cols };
            let w = floats(w);
            w.iter().enumerate().map(|(i, v)| v * s[at(i)]).collect()
        }
        "int8.convrot" => {
            text("rotation", &["hadamard"])?;
            let block = uint("rotation_size", &[4, 16, 64, 256, 1024, 4096, 16384, 65536])?;
            if cols % block != 0 {
                return Err(fail(format!(
                    "{cols} columns are not whole blocks of {block}"
                )));
            }
            let w = part("weight", &[I8], Some(shape), 0)?.bytes;
            let s = floats(part("scale", &[scale_dtype()?], None, rows as u64)?);
            let unit = pow2(-(block.trailing_zeros() as i32 / 2));
            let mut out = Vec::with_capacity(rows * cols);
            for (g, group) in w.chunks(block).enumerate() {
                let row = g * block / cols;
                let mut h: Vec<i32> = group.iter().map(|b| *b as i8 as i32).collect();
                hadamard(&mut h);
                out.extend(h.iter().map(|v| *v as f32 * s[row] * unit));
            }
            out
        }
        "mxfp8" => {
            uint("block_size", &[32])?;
            let blocks = cols.div_ceil(32);
            let (stored, tiled) = scales(
                text("scale_layout", &["row_major", "cublas_128x4"])?,
                blocks,
            );
            let s = floats(part("scale", &[F8_E8M0], Some(&stored), 0)?);
            let w = floats(part("weight", &fp8, Some(shape), 0)?);
            let at = |i: usize| scale_at(tiled, blocks, i / cols, i % cols / 32);
            w.iter().enumerate().map(|(i, v)| v * s[at(i)]).collect()
        }
        "nvfp4" => {
            uint("block_size", &[16])?;
            if cols % 16 != 0 {
                return Err(fail(format!("{cols} columns are not whole blocks of 16")));
            }
            let (stored, tiled) = scales(
                text("scale_layout", &["row_major", "cublas_128x4"])?,
                cols / 16,
            );
            let mut w = part("weight", &[F4], Some(shape), 0)?;
            let swapped: Vec<u8>;
            if text("nibble_order", &["low_first", "high_first"])? == "high_first" {
                swapped = w.bytes.iter().map(|b| b.rotate_left(4)).collect();
                w.bytes = &swapped;
            }
            let s = floats(part("scale", &[F8_E4M3], Some(&stored), 0)?);
            let g = floats(part("scale_2", &[F32], None, 1)?)[0];
            if parts.contains_key("input_scale") {
                part("input_scale", &[F32], None, 1)?;
            }
            let multiply = text("global_scale", &["multiply", "divide"])? == "multiply";
            let w = floats(w);
            let block = |i: usize| s[scale_at(tiled, cols / 16, i / cols, i % cols / 16)];
            match multiply {
                true => w
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * block(i) * g)
                    .collect(),
                false => w
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * (block(i) / g))
                    .collect(),
            }
        }
        _ => unreachable!("matched above"),
    };
    Ok(round(dtype, &out))
}

/// Where row `i`'s scale for block `b` sits, `blocks` to a row: row-major, or in cuBLAS's tiles
/// of 128 rows by 4 blocks (spec §6).
fn scale_at(tiled: bool, blocks: usize, i: usize, b: usize) -> usize {
    match tiled {
        false => i * blocks + b,
        true => {
            let tile = (i / 128) * blocks.div_ceil(4) + b / 4;
            tile * 512 + (i % 32) * 16 + (i % 128 / 32) * 4 + b % 4
        }
    }
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
        Dtype::BF16 => (b.as_chunks::<2>().0.iter())
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        Dtype::F16 => (b.as_chunks::<2>().0.iter())
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
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
    if byte & sign_bit != 0 { -v } else { v }
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
    let v = match (e, m) {
        (0, _) => m as f32 * pow2(-24),
        (31, 0) => f32::INFINITY,
        (31, _) => f32::NAN,
        _ => (1024 + m) as f32 * pow2(e - 25),
    };
    signed((h >> 8) as u8, 0x80, v)
}
