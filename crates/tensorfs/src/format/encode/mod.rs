//! The writer side of each scheme (spec §6): its complete `Encoding`, and a registry row with
//! its encoders. `quantize` is a closed form TensorFS runs itself (RTN, MX/NV FP, GGUF via
//! ggml); `pack` takes the codes, scales, zeros and permutation a calibrated producer computed
//! (GPTQ, AWQ, SVDQuant) and lays them out as the scheme's parts. Shared math: `elem` (f32 to
//! element codes) and `affine` (codes to packed words, the inverse of the decode's unpack).
//!
//! Adding a scheme's encoders: one family file, one row in `ENCODERS` (and its `mod` line).

use std::collections::BTreeMap;

use super::decode::PartBytes;
use super::{Dtype, Encoding, Error, Result, Value};

pub mod affine;
pub mod elem;

mod fp8;
mod int8;
mod mxfp8;
mod nvfp4;

// fp8.blockwise, mxfp4
mod fp8_blockwise;
mod mxfp4;

// gguf
mod gguf;

// gptq, awq, ct
mod awq;
mod ct;
mod gptq;

// bnb
mod bnb;

// comfy
mod comfy;

pub use comfy::{quantize_with, Kitchen};
pub use nvfp4::{GlobalScale, NibbleOrder};

/// Every scheme's encoders, one row each.
pub static ENCODERS: &[&Encoder] = &[
    &nvfp4::NVFP4,

    // fp8.blockwise, mxfp4
    &fp8_blockwise::ENCODER,
    &mxfp4::ENCODER,

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

/// An encoder: the parts of `input.encoding`, in the scheme's part order.
pub type Encode = fn(&Input) -> Result<Vec<OwnedPart>>;

pub struct Encoder {
    pub scheme: &'static str,
    /// From the logical values: `arrays["values"]`, `F32` in the logical shape.
    pub quantize: Option<Encode>,
    /// From a calibrated producer's arrays, by the producer's names.
    pub pack: Option<Encode>,
}

pub struct Input<'a> {
    /// The logical tensor's dtype and shape.
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    /// The complete encoding to produce: every param its decode reads.
    pub encoding: &'a Encoding,
    pub arrays: &'a BTreeMap<&'a str, PartBytes<'a>>,
}

impl Input<'_> {
    /// `arrays["values"]`: the logical values, `F32` in the logical shape, all finite.
    pub fn values(&self) -> Result<Vec<f32>> {
        let fail = |d: &str| Error::Encode(format!("{}: {d}", self.encoding.scheme));
        let v = (self.arrays.get("values")).ok_or_else(|| fail("no values"))?;
        if *v.dtype != Dtype::F32 || v.shape != self.shape {
            return Err(fail("values are not F32 in the logical shape"));
        }
        let x = super::decode::elem::floats(v.dtype, v.bytes);
        if x.iter().any(|x| !x.is_finite()) {
            return Err(fail("a value is not finite"));
        }
        Ok(x)
    }
}

/// One encoded part, in the scheme's part order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedPart {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

fn row(scheme: &str) -> Result<&'static Encoder> {
    ENCODERS
        .iter()
        .copied()
        .find(|e| e.scheme == scheme)
        .ok_or_else(|| Error::UnknownScheme(scheme.into()))
}

fn run(
    input: &Input,
    f: Option<Encode>,
    what: &str,
) -> Result<Vec<OwnedPart>> {
    let scheme = &input.encoding.scheme;
    let f = f.ok_or_else(|| Error::Encode(format!("{scheme}: no {what} encoder")))?;
    f(input)
}

/// The parts of `input.encoding` from the logical values (closed form).
pub fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    run(input, row(&input.encoding.scheme)?.quantize, "closed-form")
}

/// The parts of `input.encoding` from a calibrated producer's arrays.
pub fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    run(input, row(&input.encoding.scheme)?.pack, "pack")
}

/// An `Encoding` from its scheme and params.
pub(crate) fn encoding(scheme: &str, params: &[(&str, Value)]) -> Encoding {
    Encoding {
        scheme: scheme.into(),
        params: params
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

/// An integer array (`I8`, `U8`, `I16`, `U16` or `I32`) of a calibrated producer's, by name.
fn array(input: &Input, role: &str) -> Result<Vec<i32>> {
    let fail = |d: &str| Error::Encode(format!("{}: {role} {d}", input.encoding.scheme));
    let a = input.arrays.get(role).ok_or_else(|| fail("is missing"))?;
    let b = a.bytes;
    Ok(match a.dtype {
        Dtype::I8 => b.iter().map(|v| *v as i8 as i32).collect(),
        Dtype::U8 => b.iter().map(|v| *v as i32).collect(),
        Dtype::I16 => (b.as_chunks::<2>().0.iter())
            .map(|v| i16::from_le_bytes(*v) as i32)
            .collect(),
        Dtype::U16 => (b.as_chunks::<2>().0.iter())
            .map(|v| u16::from_le_bytes(*v) as i32)
            .collect(),
        Dtype::I32 => (b.as_chunks::<4>().0.iter())
            .map(|v| i32::from_le_bytes(*v))
            .collect(),
        other => return Err(fail(&format!("is {other}, not an integer array"))),
    })
}

fn part(
    role: &str,
    dtype: Dtype,
    shape: impl IntoIterator<Item = usize>,
    bytes: Vec<u8>,
) -> OwnedPart {
    OwnedPart {
        role: role.into(),
        dtype,
        shape: shape.into_iter().map(|e| e as u64).collect(),
        bytes,
    }
}
