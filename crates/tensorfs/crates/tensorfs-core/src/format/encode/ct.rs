//! `ct.int1` … `ct.int8`: `pack` lays codes, scales, zero points and an act-order `g_idx` out as
//! compressed-tensors' `pack-quantized` parts; `quantize` is llm-compressor's round to nearest bit
//! for bit (compressed-tensors' `calculate_qparams` on each group's min/max, then its quantize,
//! every operation in the logical dtype as torch evaluates it: f32, then rounded).

use super::affine::pack as packed;
use super::{array, encoding, part, Encoder, Input, OwnedPart};
use crate::format::decode::affine::Fill;
use crate::format::decode::ct::words;
use crate::format::decode::{bf16, elem, f16};
use crate::format::{Dtype, Encoding, Error, Result, Value};

pub const INT1: Encoder = row("ct.int1");
pub const INT2: Encoder = row("ct.int2");
pub const INT3: Encoder = row("ct.int3");
pub const INT4: Encoder = row("ct.int4");
pub const INT5: Encoder = row("ct.int5");
pub const INT6: Encoder = row("ct.int6");
pub const INT7: Encoder = row("ct.int7");
pub const INT8: Encoder = row("ct.int8");

const fn row(scheme: &'static str) -> Encoder {
    Encoder {
        scheme,
        quantize: Some(quantize),
        pack: Some(pack),
    }
}

impl Encoding {
    /// `strategy` `channel` takes `group_size` = the row length; `packing` `dense` is
    /// compressed-tensors' since 0.17, `word` before.
    pub fn ct(
        bits: u32,
        group_size: u64,
        strategy: &str,
        symmetric: bool,
        packing: &str,
    ) -> Encoding {
        let p = [
            ("group_size", group_size.into()),
            ("strategy", strategy.into()),
            ("symmetric", symmetric.into()),
            ("packing", packing.into()),
        ];
        encoding(&format!("ct.int{bits}"), &p)
    }
}

/// The encoding's code width, fill, group size and symmetry, and the view `[R, C]` with `N` the
/// rows of each matrix.
struct Layout {
    bits: u32,
    fill: Fill,
    g: usize,
    sym: bool,
    rows: usize,
    cols: usize,
    n: usize,
}

fn layout(input: &Input) -> Result<Layout> {
    let e = input.encoding;
    let fail = |d: &str| Error::Encode(format!("{}: {d}", e.scheme));
    let bits: u32 = (e.scheme.strip_prefix("ct.int").and_then(|b| b.parse().ok()))
        .filter(|b| (1..=8).contains(b))
        .ok_or_else(|| fail("not ct.int1 … ct.int8"))?;
    let [.., n, c] = *input.shape else {
        return Err(fail("the logical shape is not rank 2 or more"));
    };
    let fill = match e.params.get("packing").and_then(Value::as_str) {
        Some("word") => Fill::Word,
        Some("dense") => Fill::Dense,
        _ => return Err(fail("packing is not word or dense")),
    };
    let g = e
        .params
        .get("group_size")
        .and_then(Value::as_u64)
        .filter(|g| *g > 0);
    let g = g.ok_or_else(|| fail("no group_size"))? as usize;
    let rows = input.shape.iter().product::<u64>() as usize / (c as usize).max(1);
    let sym = e.params.get("symmetric") == Some(&Value::Bool(true));
    Ok(Layout {
        bits,
        fill,
        g,
        sym,
        rows,
        cols: c as usize,
        n: n as usize,
    })
}

/// From `q` (signed codes, the logical shape), `scales` (`[lead…, N, W]`), `zeros` (signed, the
/// same shape; asymmetric only) and, for a group act order, `g_idx` (`I32 [C]`).
fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    let Some(scales) = input.arrays.get("scales") else {
        return Err(Error::Encode(format!(
            "{}: no scales",
            input.encoding.scheme
        )));
    };
    let l = layout(input)?;
    let zeros = match l.sym {
        true => None,
        false => Some(array(input, "zeros")?),
    };
    let scales = (scales.dtype.clone(), scales.bytes.to_vec());
    let g_idx = input.arrays.get("g_idx").map(|g| g.bytes.to_vec());
    parts(input, &l, &array(input, "q")?, zeros, scales, g_idx)
}

fn parts(
    input: &Input,
    l: &Layout,
    q: &[i32],
    zeros: Option<Vec<i32>>,
    (dtype, scales): (Dtype, Vec<u8>),
    g_idx: Option<Vec<u8>>,
) -> Result<Vec<OwnedPart>> {
    let shape = input.shape;
    let lead: Vec<usize> = shape[..shape.len() - 1]
        .iter()
        .map(|e| *e as usize)
        .collect();
    let w = l.cols.div_ceil(l.g);
    let along_k = words(l.bits, l.fill, 1);
    let mut out = vec![
        part(
            "weight_packed",
            Dtype::I32,
            [&lead[..], &[along_k.words(l.cols)?]].concat(),
            packed(&along_k, q, l.rows, l.cols)?,
        ),
        part("weight_scale", dtype, [&lead[..], &[w]].concat(), scales),
    ];
    if let Some(z) = zeros {
        let along_n = words(l.bits, l.fill, 0);
        let mut bytes = Vec::new();
        for m in z.chunks((l.n * w).max(1)) {
            bytes.extend(packed(&along_n, m, l.n, w)?);
        }
        let matrices = &lead[..lead.len() - 1];
        out.push(part(
            "weight_zero_point",
            Dtype::I32,
            [matrices, &[along_n.words(l.n)?, w]].concat(),
            bytes,
        ));
    }
    let stated = shape
        .iter()
        .flat_map(|e| (*e as i64).to_le_bytes())
        .collect();
    out.push(part("weight_shape", Dtype::I64, [shape.len()], stated));
    if let Some(g_idx) = g_idx {
        out.push(part("weight_g_idx", Dtype::I32, [l.cols], g_idx));
    }
    Ok(out)
}

/// Rounds an f32 to the logical dtype.
type Round = fn(f32) -> f32;

/// The logical dtype's rounding and machine epsilon.
fn rounding(dtype: &Dtype) -> Result<(Round, f32)> {
    Ok(match dtype {
        Dtype::BF16 => (|v| elem::bf16(bf16(v)), 2f32.powi(-7)),
        Dtype::F16 => (|v| elem::f16(f16(v)), 2f32.powi(-10)),
        Dtype::F32 => (|v| v, f32::EPSILON),
        other => return Err(Error::Encode(format!("ct: logical dtype {other}"))),
    })
}

/// Per group of a row, in the logical dtype `L` (each operation f32, then rounded to `L`):
/// `lo = min(x, 0)`, `hi = max(x, 0)`; symmetric `s = max(|lo|, |hi|) / ((2^b − 1)/2)`, `zp = 0`;
/// asymmetric `s = (hi − lo) / (2^b − 1)`, `zp = round(clamp(−2^(b−1) − lo/s))`; a zero `s` is
/// `L`'s epsilon (and its group's `zp` 0); codes `round(clamp(x/s + zp))`, ties to even.
fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let l = layout(input)?;
    let (r, eps) = rounding(input.dtype)?;
    let x: Vec<f32> = input.values()?.into_iter().map(r).collect();
    let (qmin, qmax) = (
        -((1 << (l.bits - 1)) as f32),
        ((1 << (l.bits - 1)) - 1) as f32,
    );
    let w = l.cols.div_ceil(l.g);
    let (mut q, mut zeros, mut scales) = (vec![0; x.len()], vec![0; l.rows * w], Vec::new());
    for i in 0..l.rows {
        for b in 0..w {
            let cols = b * l.g..((b + 1) * l.g).min(l.cols);
            let group = &x[i * l.cols..][cols.clone()];
            let lo = group.iter().fold(0f32, |a, v| a.min(*v));
            let hi = group.iter().fold(0f32, |a, v| a.max(*v));
            let (s, zp) = match l.sym {
                true => (r(lo.abs().max(hi) / ((qmax - qmin) / 2.0)), 0.0),
                false => {
                    let s = r(r(hi - lo) / (qmax - qmin));
                    let zp = r(qmin - r(lo / s)).clamp(qmin, qmax).round_ties_even();
                    (s, if s == 0.0 { 0.0 } else { zp })
                }
            };
            let s = if s == 0.0 { eps } else { s };
            for (j, v) in cols.zip(group) {
                q[i * l.cols + j] = r(r(v / s) + zp).clamp(qmin, qmax).round_ties_even() as i32;
            }
            zeros[i * w + b] = zp as i32;
            scales.extend(match input.dtype {
                Dtype::BF16 => bf16(s).to_le_bytes().to_vec(),
                Dtype::F16 => f16(s).to_le_bytes().to_vec(),
                _ => s.to_le_bytes().to_vec(),
            });
        }
    }
    let zeros = (!l.sym).then_some(zeros);
    parts(input, &l, &q, zeros, (input.dtype.clone(), scales), None)
}
