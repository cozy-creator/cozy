//! The reference CPU decode (spec §6): a registry with one row per scheme, the checks every
//! row shares, and the cores schemes are built from (`elem`, `grid`, `affine`, `blocks`). A row
//! returns its decode as a `plan`: here each element is computed in f32 by the scheme's formula,
//! then rounded once to the logical dtype (`round_L`); TensorFS's GPU kernels run the same plan.
//!
//! Adding a scheme: one family file holding its `Scheme` row and decode fn, one row in
//! `SCHEMES` (and its `mod` line), nothing else here.

use std::collections::BTreeMap;
use std::fmt::Display;

use super::{Dtype, Encoding, Error, Result, Value};
pub use plan::Plan;

pub mod affine;
pub mod blocks;
pub mod elem;
pub mod grid;
pub mod plan;

mod fp8;
mod int8;
mod mxfp8;
mod nvfp4;

// fp8.blockwise, mxfp4
mod fp8_blockwise;
mod mxfp4;

// gguf
pub mod gguf;

// gptq, awq, ct
pub(crate) mod awq;
pub(crate) mod ct;
pub(crate) mod gptq;

// bnb
pub(crate) mod bnb;

// comfy
mod comfy;

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
    &fp8_blockwise::BLOCKWISE,
    &mxfp4::MXFP4,
    // gguf.*
    &gguf::Q4_0,
    &gguf::Q4_1,
    &gguf::Q5_0,
    &gguf::Q5_1,
    &gguf::Q8_0,
    &gguf::Q2_K,
    &gguf::Q3_K,
    &gguf::Q4_K,
    &gguf::Q5_K,
    &gguf::Q6_K,
    &gguf::MXFP4,
    &gguf::NVFP4,
    &gguf::Q1_0,
    &gguf::Q2_0,
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
    &bnb::NF4,
    &bnb::FP4,
    &bnb::INT8,
    // comfy.*
    &comfy::ASYM_W4A8_INT8,
    &comfy::W6A8_INT8,
    &comfy::CONVROT_W4A4,
];

/// A scheme: its params, its parts and its decode, as spec §6 lists them.
pub struct Scheme {
    pub name: &'static str,
    pub view: View,
    /// Every param the decode reads; all are required, none has a default.
    pub params: &'static [Param],
    pub parts: &'static [Part],
    /// The decode's plan: the CPU computes its f32 elements, row-major, and `decode` rounds them
    /// to the logical dtype; the GPU kernels run the same plan.
    pub decode: for<'a> fn(&Ctx<'a>) -> Result<Plan<'a>>,
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

/// A part's allowed dtypes, by name: the part's own dtype is authoritative.
pub type Dtypes = &'static [&'static str];

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

/// Stored part metadata, independent of whether its bytes live on the Host or a device.
#[derive(Clone, Copy, Debug)]
pub struct PartMetadata<'a> {
    pub role: &'a str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub len: usize,
}

type Slots<'a> = BTreeMap<&'a str, (usize, usize)>;

/// What a decode fn sees: the view, the checked params and the checked parts.
pub struct Ctx<'a> {
    pub scheme: &'static str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub rows: usize,
    pub cols: usize,
    params: &'a BTreeMap<String, Value>,
    parts: &'a BTreeMap<&'a str, PartBytes<'a>>,
    slots: Option<&'a Slots<'a>>,
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
    /// The SAME checked scheme mapping can refer to an engine-bound input slot.
    pub(crate) fn source(&self, role: &str) -> plan::Bytes<'a> {
        match self.slots.and_then(|s| s.get(role)) {
            Some(&(index, len)) => plan::Bytes::slot(index, len),
            None => self.part(role).bytes.into(),
        }
    }
    pub(crate) fn elements(&self, role: &str) -> plan::Elems<'a> {
        plan::Elems {
            dtype: self.part(role).dtype.clone(),
            bytes: self.source(role),
        }
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
    /// The rows of one matrix: the second-last extent of a matrix view, else `R`.
    pub fn height(&self) -> usize {
        match self.shape {
            [.., n, _] if self.cols > 0 && self.rows > 0 && self.shape.len() >= 2 => *n as usize,
            _ => self.rows,
        }
        .max(1)
    }
}

/// Where a plan's values go: the `[R, C]` view with `height` rows to a matrix. `swap`: the parts
/// hold the logical tensor with its last two axes swapped (`transposed: true`), so the values are
/// swapped back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub rows: usize,
    pub cols: usize,
    pub height: usize,
    pub swap: bool,
}

/// The logical tensor's bytes from its encoded parts.
pub fn decode(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
) -> Result<Vec<u8>> {
    planned(dtype, shape, encoding, parts, true, |plan, g| {
        let out = plan.values(g.rows, g.cols, g.height)?;
        let out = match g.swap {
            true => swap_last_two(&out, g.height, g.cols),
            false => out,
        };
        Ok(round(dtype, &out))
    })
}

/// Refuses what `decode` would refuse before reading a byte: the scheme, its params, the logical
/// dtype and shape, and each part's role, dtype and shape (the parts' `bytes` are not read).
pub fn check(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
) -> Result<()> {
    planned(dtype, shape, encoding, parts, false, |_, _| Ok(()))
}

/// `f` with the tensor's checked plan (built when `read`, else `Values` of nothing) and its
/// geometry; a refusal names the scheme.
pub fn planned<R>(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &BTreeMap<&str, PartBytes>,
    read: bool,
    f: impl FnOnce(Plan, Geometry) -> Result<R>,
) -> Result<R> {
    let name = encoding.scheme.as_str();
    let Some(scheme) = SCHEMES.iter().find(|s| s.name == name) else {
        return Err(Error::UnknownScheme(name.into()));
    };
    run(
        scheme,
        (dtype, shape),
        &encoding.params,
        parts,
        None,
        read,
        f,
    )
    .map_err(|e| match e {
        Error::Decode(detail) => Error::Decode(format!("{name}: {detail}")),
        e => e,
    })
}

/// Metadata preflight using the existing registry; no data is read or device initialized.
pub fn check_metadata(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &[PartMetadata<'_>],
) -> Result<()> {
    inputs(dtype, shape, encoding, parts, false, |_, _| Ok(()))
}

#[cfg(feature = "store")]
pub(crate) fn supports_sources(encoding: &Encoding) -> bool {
    SCHEMES
        .iter()
        .any(|scheme| scheme.name == encoding.scheme && comfy::supports_sources(scheme))
}

/// Plan external sources with the SAME U7 Scheme::decode functions used by Host reads.
#[cfg(feature = "store")]
pub(crate) fn planned_sources<R>(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &[PartMetadata<'_>],
    f: impl FnOnce(Plan, Geometry) -> Result<R>,
) -> Result<R> {
    inputs(dtype, shape, encoding, parts, true, f)
}

fn inputs<R>(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &[PartMetadata<'_>],
    read: bool,
    f: impl FnOnce(Plan, Geometry) -> Result<R>,
) -> Result<R> {
    let name = encoding.scheme.as_str();
    let scheme = SCHEMES
        .iter()
        .find(|s| s.name == name)
        .copied()
        .ok_or_else(|| Error::UnknownScheme(name.into()))?;
    if read && !comfy::supports_sources(scheme) {
        return Err(Error::Decode(format!(
            "{name}: resident sources have no device plan"
        )));
    }
    let (mut metadata, mut slots) = (BTreeMap::new(), BTreeMap::new());
    for (index, p) in parts.iter().enumerate() {
        if metadata
            .insert(
                p.role,
                PartBytes {
                    dtype: p.dtype,
                    shape: p.shape,
                    bytes: &[],
                },
            )
            .is_some()
        {
            return Err(Error::Decode(format!(
                "{name}: duplicate part {:?}",
                p.role
            )));
        }
        slots.insert(p.role, (index, p.len));
    }
    run(
        scheme,
        (dtype, shape),
        &encoding.params,
        &metadata,
        Some(&slots),
        read,
        f,
    )
    .map_err(|e| match e {
        Error::Decode(d) => Error::Decode(format!("{name}: {d}")),
        e => e,
    })
}

fn run<R>(
    scheme: &'static Scheme,
    logical: (&Dtype, &[u64]),
    params: &BTreeMap<String, Value>,
    parts: &BTreeMap<&str, PartBytes>,
    slots: Option<&Slots>,
    read: bool,
    f: impl FnOnce(Plan, Geometry) -> Result<R>,
) -> Result<R> {
    let (dtype, shape) = logical;
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
    // `transposed: true` (spec §6, building blocks): the parts hold the logical tensor with its
    // last two axes swapped; the scheme reads that stored tensor, and its values are swapped back.
    let transposed = matches!(params.get("transposed"), Some(Value::Bool(true)));
    let stored: Vec<u64> = match (transposed, shape) {
        (true, [lead @ .., a, b]) => [lead, &[*b, *a]].concat(),
        _ => shape.to_vec(),
    };
    let shape = stored.as_slice();
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
            )));
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
        slots,
    };
    for rule in scheme.parts {
        let Some(p) = parts.get(rule.role) else {
            continue;
        };
        if !rule.dtypes.contains(&p.dtype.name()) {
            let allowed = rule.dtypes.join(", ");
            return Err(fail(format!("part {} is {}, not {allowed}", rule.role, p.dtype)));
        }
        let fits = match (rule.shape)(&ctx)? {
            Shape::Exact(want) => p.shape == want.as_slice(),
            Shape::Count(n) => p.shape.iter().try_fold(1u64, |a, b| a.checked_mul(*b)) == Some(n),
        };
        let len = slots
            .and_then(|s| s.get(rule.role))
            .map(|(_, len)| *len)
            .or_else(|| read.then_some(p.bytes.len()));
        if !fits || len.is_some_and(|len| p.dtype.byte_len(p.shape) != Some(len as u64)) {
            return Err(fail(format!("part {} has shape {:?}", rule.role, p.shape)));
        }
    }
    let g = Geometry {
        rows,
        cols,
        height: ctx.height(),
        swap: transposed,
    };
    let plan = match read {
        true => (scheme.decode)(&ctx)?,
        false => Plan {
            core: plan::Core::Values(Vec::new()),
            post: plan::Post::None,
        },
    };
    f(plan, g)
}

/// Values of matrices of `r × c`, row-major, as `c × r`.
pub fn swap_last_two(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    if r * c == 0 {
        return x.to_vec();
    }
    let mut out = x.to_vec();
    for (m, mat) in x.chunks(r * c).enumerate() {
        for (k, v) in mat.iter().enumerate() {
            out[m * r * c + (k % c) * r + k / c] = *v;
        }
    }
    out
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
pub(crate) fn round(dtype: &Dtype, xs: &[f32]) -> Vec<u8> {
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
