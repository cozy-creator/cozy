//! `gptq.int{2,3,4,8}`: `pack` lays a calibrated GPTQ's codes, zeros, scales and `g_idx` out as
//! GPTQ's own parts (GPTQModel's and AutoGPTQ's bytes, v1 or v2 zeros); `quantize` is round to
//! nearest on GPTQ's min/max grid, in column order with `F16` scales.

use super::affine::pack as packed;
use super::{array, encoding, part, Encoder, Input, OwnedPart};
use crate::format::decode::gptq::{qweight, qzeros};
use crate::format::decode::{elem, f16};
use crate::format::{Dtype, Encoding, Error, Result, Value};

pub const INT2: Encoder = row("gptq.int2");
pub const INT3: Encoder = row("gptq.int3");
pub const INT4: Encoder = row("gptq.int4");
pub const INT8: Encoder = row("gptq.int8");

const fn row(scheme: &'static str) -> Encoder {
    Encoder {
        scheme,
        quantize: Some(quantize),
        pack: Some(pack),
    }
}

impl Encoding {
    /// `group_size` `-1` is one group per row; `zero_offset` 1 is GPTQ's v1 zeros, 0 its v2.
    pub fn gptq(
        bits: u32,
        group_size: i64,
        symmetric: bool,
        act_order: bool,
        zero_offset: u64,
    ) -> Encoding {
        let p = [
            ("group_size", group_size.into()),
            ("symmetric", symmetric.into()),
            ("act_order", act_order.into()),
            ("zero_offset", zero_offset.into()),
        ];
        encoding(&format!("gptq.int{bits}"), &p)
    }
}

/// A GPTQ-family encoding's code width, `[N, K]` and group size along K (`-1`: all of K).
pub(crate) fn layout(input: &Input) -> Result<(u32, usize, usize, usize)> {
    let e = input.encoding;
    let fail = |d: &str| Error::Encode(format!("{}: {d}", e.scheme));
    let bits = (e
        .scheme
        .rsplit_once(".int")
        .and_then(|(_, b)| b.parse().ok()))
    .ok_or_else(|| fail("not an intB scheme"))?;
    let [n, k] = input.shape[..] else {
        return Err(fail("the logical shape is not [N, K]"));
    };
    let g = match e.params.get("group_size") {
        Some(Value::Int(-1)) => k.max(1),
        Some(v) => v
            .as_u64()
            .filter(|g| *g > 0)
            .ok_or_else(|| fail("bad group_size"))?,
        None => return Err(fail("no group_size")),
    };
    Ok((bits, n as usize, k as usize, g as usize))
}

fn flag(input: &Input, key: &str) -> bool {
    input.encoding.params.get(key) == Some(&Value::Bool(true))
}

/// From `q` (`[N, K]`), `zeros` (`[W, N]`, the zeros themselves), `scales` (`F16` or `BF16`
/// `[W, N]`) and `g_idx` (`I32 [K]`): `qweight`, `qzeros` (each slot `z − zero_offset`, modulo
/// `2^b`), `scales` and `g_idx`.
fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    let (scales, g_idx) = (input.arrays.get("scales"), input.arrays.get("g_idx"));
    let (Some(scales), Some(g_idx)) = (scales, g_idx) else {
        return Err(Error::Encode(format!(
            "{}: no scales or g_idx",
            input.encoding.scheme
        )));
    };
    let scales = (scales.dtype.clone(), scales.bytes.to_vec());
    let (q, z) = (array(input, "q")?, array(input, "zeros")?);
    parts(input, &q, &z, scales, g_idx.bytes.to_vec())
}

fn parts(
    input: &Input,
    q: &[i32],
    zeros: &[i32],
    (dtype, scales): (Dtype, Vec<u8>),
    g_idx: Vec<u8>,
) -> Result<Vec<OwnedPart>> {
    let (bits, n, k, g) = layout(input)?;
    let w = k.div_ceil(g);
    let offset = match input.encoding.params.get("zero_offset") {
        Some(Value::Int(1)) => 1,
        _ => 0,
    };
    let mask = (1 << bits) - 1;
    let stored: Vec<i32> = zeros.iter().map(|z| (z - offset) & mask).collect();
    let (pw, pz) = (qweight(bits), qzeros(bits));
    Ok(vec![
        part(
            "qweight",
            Dtype::I32,
            [pw.words(k)?, n],
            packed(&pw, q, n, k)?,
        ),
        part(
            "qzeros",
            Dtype::I32,
            [w, pz.words(n)?],
            packed(&pz, &stored, w, n)?,
        ),
        part("scales", dtype, [w, n], scales),
        part("g_idx", Dtype::I32, [k], g_idx),
    ])
}

/// GPTQ's min/max grid per group of a row (zero kept in range; symmetric: `±max|x|` and the zero
/// `2^(b−1)`), the scale stored as `F16`, codes `clamp(round(x / s) + z)` against the stored
/// scale, ties to even. Column order only: an activation order comes from calibration.
fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let (bits, n, k, g) = layout(input)?;
    if flag(input, "act_order") {
        let e = &input.encoding.scheme;
        return Err(Error::Encode(format!(
            "{e}: an activation order needs calibration"
        )));
    }
    let sym = flag(input, "symmetric");
    let x = input.values()?;
    let w = k.div_ceil(g);
    let maxq = ((1 << bits) - 1) as f32;
    let (mut q, mut zeros, mut scales) = (vec![0; n * k], vec![0; w * n], Vec::new());
    for b in 0..w {
        let cols = b * g..((b + 1) * g).min(k);
        for i in 0..n {
            let group = &x[i * k..][cols.clone()];
            let mut lo = group.iter().fold(0f32, |a, v| a.min(*v));
            let mut hi = group.iter().fold(0f32, |a, v| a.max(*v));
            if sym {
                hi = hi.max(-lo);
                lo = if lo < 0.0 { -hi } else { lo };
            }
            if lo == 0.0 && hi == 0.0 {
                (lo, hi) = (-1.0, 1.0);
            }
            let scale = (hi - lo) / maxq;
            let stored = f16(scale).max(1);
            let s = elem::f16(stored);
            let z = match sym {
                true => (1 << (bits - 1)) as f32,
                false => (-lo / scale).round_ties_even().clamp(0.0, maxq),
            };
            for (j, v) in cols.clone().zip(group) {
                q[i * k + j] = ((v / s).round_ties_even() + z).clamp(0.0, maxq) as i32;
            }
            zeros[b * n + i] = z as i32;
            scales.extend(stored.to_le_bytes());
        }
    }
    let g_idx = (0..k)
        .flat_map(|j| ((j / g) as i32).to_le_bytes())
        .collect();
    parts(input, &q, &zeros, (Dtype::F16, scales), g_idx)
}
