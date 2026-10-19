//! `awq.int4`: `pack` lays a calibrated AWQ's codes, zeros and scales out in AutoAWQ's GEMM layout
//! (its bytes and GPTQModel's); `quantize` is round to nearest on AutoAWQ's min/max grid (zero
//! point, `F16` scales). AWQ's activation scaling is folded into the previous op by its producer.

use super::affine::pack as packed;
use super::gptq::layout;
use super::{array, encoding, part, Encoder, Input, OwnedPart};
use crate::format::decode::awq::{CODES, ZEROS};
use crate::format::decode::{elem, f16};
use crate::format::{Dtype, Encoding, Error, Result};

pub const INT4: Encoder = Encoder {
    scheme: "awq.int4",
    quantize: Some(quantize),
    pack: Some(pack),
};

impl Encoding {
    pub fn awq_int4(group_size: i64) -> Encoding {
        encoding("awq.int4", &[("group_size", group_size.into())])
    }
}

/// From `q` (`[N, K]`), `zeros` (`[W, N]`) and `scales` (`F16` or `BF16` `[W, N]`).
fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    let Some(scales) = input.arrays.get("scales") else {
        return Err(Error::Encode("awq.int4: no scales".into()));
    };
    let scales = (scales.dtype.clone(), scales.bytes.to_vec());
    parts(input, &array(input, "q")?, &array(input, "zeros")?, scales)
}

fn parts(
    input: &Input,
    q: &[i32],
    z: &[i32],
    (dtype, scales): (Dtype, Vec<u8>),
) -> Result<Vec<OwnedPart>> {
    let (_, n, k, g) = layout(input)?;
    let w = k.div_ceil(g);
    Ok(vec![
        part(
            "qweight",
            Dtype::I32,
            [k, CODES.words(n)?],
            packed(&CODES, q, n, k)?,
        ),
        part(
            "qzeros",
            Dtype::I32,
            [w, ZEROS.words(n)?],
            packed(&ZEROS, z, w, n)?,
        ),
        part("scales", dtype, [w, n], scales),
    ])
}

/// Per group of a row: `s = max(hi − lo, 10⁻⁵) / 15` stored as `F16`, `z = clamp(−round(lo / s))`,
/// codes `clamp(round(x / s) + z)` against the stored scale, ties to even.
fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let (_, n, k, g) = layout(input)?;
    let x = input.values()?;
    let w = k.div_ceil(g);
    let (mut q, mut zeros, mut scales) = (vec![0; n * k], vec![0; w * n], Vec::new());
    for b in 0..w {
        let cols = b * g..((b + 1) * g).min(k);
        for i in 0..n {
            let group = &x[i * k..][cols.clone()];
            let lo = group.iter().fold(f32::INFINITY, |a, v| a.min(*v));
            let hi = group.iter().fold(f32::NEG_INFINITY, |a, v| a.max(*v));
            let scale = (hi - lo).max(1e-5) / 15.0;
            let stored = f16(scale).max(1);
            let s = elem::f16(stored);
            let z = (-(lo / scale).round_ties_even()).clamp(0.0, 15.0);
            for (j, v) in cols.clone().zip(group) {
                q[i * k + j] = ((v / s).round_ties_even() + z).clamp(0.0, 15.0) as i32;
            }
            zeros[b * n + i] = z as i32;
            scales.extend(stored.to_le_bytes());
        }
    }
    parts(input, &q, &zeros, (Dtype::F16, scales))
}
