//! The reference CPU decode (spec §6): a registry with one row per scheme, the checks every
//! row shares, and the cores schemes are built from (`elem`, `grid`, `affine`, `table`,
//! `blocks`). Each element is computed in f32 by the scheme's formula, then rounded once to the
//! logical dtype (`round_L`). The Runtime's GPU decoders match it bit for bit.
//!
//! Adding a scheme: one family file holding its `Scheme` row and decode fn, one row in
//! `SCHEMES` (and its `mod` line), nothing else here.

use std::collections::BTreeMap;
use std::fmt::Display;

use super::{Dtype, Encoding, Error, Result, Value};

pub mod affine;
pub mod blocks;
pub mod elem;
pub mod grid;
pub mod table;

mod fp8;
mod int8;
mod mxfp8;
mod nvfp4;

// fp8.blockwise, mxfp4

// gguf

// gptq, awq, ct
mod awq;
mod ct;
mod gptq;

// bnb

// comfy

/// Every scheme this reader decodes, one row each. A plain tensor needs no scheme.
pub static SCHEMES: &[&Scheme] = &[
    &fp8::TENSORWISE,
    &fp8::ROWWISE,
    &int8::TENSORWISE,
    &int8::ROWWISE,
    &int8::CONVROT,
    &mxfp8::MXFP8,
    &nvfp4::NVFP4,
    // fp8.blockwise, mxfp4

    // gguf.*

    // gptq.*, awq.*, ct.*
    &gptq::INT2,
    &gptq::INT3,
    &gptq::INT4,
    &gptq::INT8,
    &awq::INT4,
    &ct::INT1,
    &ct::INT2,
    &ct::INT3,
    &ct::INT4,
    &ct::INT5,
    &ct::INT6,
    &ct::INT7,
    &ct::INT8,

    // bnb.*

    // comfy.*
];

/// A scheme: its params, its parts and its decode, as spec §6 lists them.
pub struct Scheme {
    pub name: &'static str,
    pub view: View,
    /// Every param the decode reads; all are required, none has a default.
    pub params: &'static [Param],
    pub parts: &'static [Part],
    /// The tensor's elements in f32, row-major; `decode` rounds them to the logical dtype.
    pub decode: fn(&Ctx) -> Result<Vec<f32>>,
}

/// How a scheme views the logical tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Any shape, as one row of every element (`R = 1`).
    Flat,
    /// Rank 2 or more as `[R, C]`: `C` the last extent, `R` the product of the others.
    Matrix,
}

pub struct Param {
    pub key: &'static str,
    pub allowed: Allowed,
}

/// The values a param may take; any other value is refused.
pub enum Allowed {
    Text(&'static [&'static str]),
    Uint(&'static [u64]),
    /// An integer in `lo..=hi`.
    Range(u64, u64),
    Bool,
    /// Values `check` accepts, as `what` describes them.
    Check(fn(&Value) -> bool, &'static str),
}

pub struct Part {
    pub role: &'static str,
    pub required: bool,
    pub dtypes: Dtypes,
    /// The stored shape this part must have, from the view and the params.
    pub shape: fn(&Ctx) -> Result<Shape>,
}

impl Part {
    pub const fn req(role: &'static str, dtypes: Dtypes, shape: fn(&Ctx) -> Result<Shape>) -> Part {
        Part {
            role,
            required: true,
            dtypes,
            shape,
        }
    }
    pub const fn opt(role: &'static str, dtypes: Dtypes, shape: fn(&Ctx) -> Result<Shape>) -> Part {
        Part {
            role,
            required: false,
            dtypes,
            shape,
        }
    }
}

/// A part's allowed dtypes, by name: listed, or the one a text param names.
pub enum Dtypes {
    Of(&'static [&'static str]),
    Param(&'static str),
}

/// A part's required stored shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    Exact(Vec<u64>),
    /// Any shape of this many elements.
    Count(u64),
}

/// The logical shape.
pub fn logical(c: &Ctx) -> Result<Shape> {
    Ok(Shape::Exact(c.shape.to_vec()))
}

/// One element, any shape.
pub fn one(_: &Ctx) -> Result<Shape> {
    Ok(Shape::Count(1))
}

/// One part's stored bytes, as the header declares them.
#[derive(Debug, Clone, Copy)]
pub struct PartBytes<'a> {
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub bytes: &'a [u8],
}

/// What a decode fn sees: the view, the checked params and the checked parts.
pub struct Ctx<'a> {
    pub scheme: &'static str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub rows: usize,
    pub cols: usize,
    params: &'a BTreeMap<String, Value>,
    parts: &'a BTreeMap<&'a str, PartBytes<'a>>,
}

impl<'a> Ctx<'a> {
    /// A refusal; `decode` prefixes the scheme's name.
    pub fn fail(&self, detail: impl Display) -> Error {
        Error::Decode(detail.to_string())
    }
    /// A text param of the row (checked before the decode runs).
    pub fn text(&self, key: &str) -> &'a str {
        self.params[key].as_str().expect("checked against the row")
    }
    pub fn uint(&self, key: &str) -> usize {
        self.params[key].as_u64().expect("checked against the row") as usize
    }
    /// A param of the row as stored, for params of no text or integer kind (checked before the
    /// decode runs).
    pub fn param(&self, key: &str) -> &'a Value {
        &self.params[key]
    }
    pub fn flag(&self, key: &str) -> bool {
        matches!(self.params[key], Value::Bool(true))
    }
    pub fn has(&self, role: &str) -> bool {
        self.parts.contains_key(role)
    }
    /// A present part (dtype and shape checked against the row).
    pub fn part(&self, role: &str) -> PartBytes<'a> {
        self.parts[role]
    }
    /// A present part's elements as f32 (`elem::floats`).
    pub fn floats(&self, role: &str) -> Vec<f32> {
        let p = self.part(role);
        elem::floats(p.dtype, p.bytes)
    }
    /// The logical shape without its last extent.
    pub fn lead(&self) -> &'a [u64] {
        &self.shape[..self.shape.len().saturating_sub(1)]
    }
}

/// The logical tensor's bytes from its encoded parts.
pub fn decode(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
) -> Result<Vec<u8>> {
    let name = encoding.scheme.as_str();
    let Some(scheme) = SCHEMES.iter().find(|s| s.name == name) else {
        return Err(Error::UnknownScheme(name.into()));
    };
    checked(scheme, dtype, shape, &encoding.params, parts).map_err(|e| match e {
        Error::Decode(detail) => Error::Decode(format!("{name}: {detail}")),
        e => e,
    })
}

fn checked(
    scheme: &'static Scheme,
    dtype: &Dtype,
    shape: &[u64],
    params: &BTreeMap<String, Value>,
    parts: &BTreeMap<&str, PartBytes>,
) -> Result<Vec<u8>> {
    let fail = Error::Decode;
    for role in parts.keys() {
        if !scheme.parts.iter().any(|p| p.role == *role) {
            return Err(fail(format!("unknown part {role:?}")));
        }
    }
    if let Some(p) = (scheme.parts.iter()).find(|p| p.required && !parts.contains_key(p.role)) {
        return Err(fail(format!("missing part {:?}", p.role)));
    }
    // Params drive the decode: one this reader does not know is a refusal, not a default.
    if let Some(key) = params
        .keys()
        .find(|k| !scheme.params.iter().any(|p| p.key == *k))
    {
        return Err(fail(format!("unknown param {key:?}")));
    }
    for p in scheme.params {
        if !params.get(p.key).is_some_and(|v| allowed(&p.allowed, v)) {
            return Err(fail(format!(
                "param {} is not {}",
                p.key,
                describe(&p.allowed)
            )));
        }
    }
    if !matches!(dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
        return Err(fail(format!(
            "logical dtype {dtype} is not F32, BF16 or F16"
        )));
    }
    let numel = shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
    let (rows, cols) = match (scheme.view, shape, numel) {
        (View::Flat, _, Some(n)) => (1, n as usize),
        (View::Matrix, [.., c], Some(n)) if shape.len() >= 2 && *c > 0 => {
            ((n / c) as usize, *c as usize)
        }
        (View::Matrix, [.., _], Some(0)) if shape.len() >= 2 => (0, 1),
        _ => {
            return Err(fail(format!(
                "logical shape {shape:?} is not rank 2 or more"
            )))
        }
    };
    let ctx = Ctx {
        scheme: scheme.name,
        dtype,
        shape,
        rows,
        cols,
        params,
        parts,
    };
    for rule in scheme.parts {
        let Some(p) = parts.get(rule.role) else {
            continue;
        };
        let ok = match rule.dtypes {
            Dtypes::Of(names) => names.contains(&p.dtype.name()),
            Dtypes::Param(key) => ctx.text(key) == p.dtype.name(),
        };
        if !ok {
            return Err(fail(format!("part {} is {}", rule.role, p.dtype)));
        }
        let fits = match (rule.shape)(&ctx)? {
            Shape::Exact(want) => p.shape == want.as_slice(),
            Shape::Count(n) => p.shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b)) == Some(n),
        };
        if !fits || p.dtype.byte_len(p.shape) != Some(p.bytes.len() as u64) {
            return Err(fail(format!("part {} has shape {:?}", rule.role, p.shape)));
        }
    }
    let out = (scheme.decode)(&ctx)?;
    debug_assert_eq!(out.len(), rows * cols);
    Ok(round(dtype, &out))
}

fn allowed(a: &Allowed, v: &Value) -> bool {
    match a {
        Allowed::Text(list) => v.as_str().is_some_and(|s| list.contains(&s)),
        Allowed::Uint(list) => v.as_u64().is_some_and(|n| list.contains(&n)),
        Allowed::Range(lo, hi) => v.as_u64().is_some_and(|n| (*lo..=*hi).contains(&n)),
        Allowed::Bool => matches!(v, Value::Bool(_)),
        Allowed::Check(check, _) => check(v),
    }
}

fn describe(a: &Allowed) -> String {
    match a {
        Allowed::Text(list) => format!("one of {list:?}"),
        Allowed::Uint(list) => format!("one of {list:?}"),
        Allowed::Range(lo, hi) => format!("an integer in {lo}..={hi}"),
        Allowed::Bool => "a boolean".into(),
        Allowed::Check(_, what) => what.to_string(),
    }
}

/// `round_L`: little-endian bytes of `xs` in `dtype` (F32, BF16 or F16), nearest even, NaN
/// canonical.
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

pub fn bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn f16(x: f32) -> u16 {
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
