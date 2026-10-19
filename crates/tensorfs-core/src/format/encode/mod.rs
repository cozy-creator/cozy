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

// gptq, awq, ct

// bnb

// comfy

pub use nvfp4::{GlobalScale, NibbleOrder};

/// Every scheme's encoders, one row each.
pub static ENCODERS: &[&Encoder] = &[
    &nvfp4::NVFP4,

    // fp8.blockwise, mxfp4
    &fp8_blockwise::ENCODER,
    &mxfp4::ENCODER,

    // gguf.*

    // gptq.*, awq.*, ct.*

    // bnb.*

    // comfy.*
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
