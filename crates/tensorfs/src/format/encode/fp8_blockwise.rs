//! `fp8.blockwise`: one scale per tile, by the convention of each scale dtype's producer, bit for
//! bit. `F32` (the registry's `quantize`) and `F8_E8M0`: transformers' FineGrainedFP8 (DeepSeek's
//! `weight_scale_inv`; `scale_fmt: ue8m0` rounds it up to a power of two). `BF16` and `F16`:
//! compressed-tensors' block strategy (llm-compressor's FP8_BLOCK). Edge tiles may be short.

use super::{elem, encoding, Encoder, Input, OwnedPart};
use crate::format::decode::{bf16, elem as exact, f16};
use crate::format::{Dtype, Encoding, Error, Result, Value};

pub const ENCODER: Encoder = Encoder {
    scheme: "fp8.blockwise",
    quantize: Some(|i| quantize_fp8_blockwise(i, &Dtype::F32)),
    pack: None,
};

impl Encoding {
    /// Tiles of `rows × cols`; the scale part is `F32`, `BF16`, `F16` or `F8_E8M0`.
    pub fn fp8_blockwise(rows: u64, cols: u64) -> Encoding {
        let size = Value::List(vec![rows.into(), cols.into()]);
        encoding("fp8.blockwise", &[("block_size", size)])
    }
}

const MAX: f32 = 448.0;

/// A 16-bit scale dtype: round to it, read it back, and its machine epsilon.
struct Half {
    to: fn(f32) -> u16,
    from: fn(u16) -> f32,
    eps: f32,
}

const BF16: Half = Half {
    to: bf16,
    from: exact::bf16,
    eps: 0.0078125,
};
const F16: Half = Half {
    to: f16,
    from: exact::f16,
    eps: 0.0009765625,
};

/// A tile's stored scale (its bytes) and its elements' codes.
fn tile(dtype: &Dtype, x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let codes = |f: &dyn Fn(f32) -> f32| {
        x.iter()
            .map(|v| elem::e4m3(f(*v).clamp(-MAX, MAX)))
            .collect()
    };
    // transformers: the forward factor 448/amax as torch evaluates it, (1/amax)·448, and the
    // stored inverse 1/forward; a zero tile stores 1.
    let forward = if amax > 0.0 { (1.0 / amax) * MAX } else { 1.0 };
    let h = match dtype {
        Dtype::F32 => {
            return (
                (1.0 / forward).to_le_bytes().to_vec(),
                codes(&|v| v * forward),
            );
        }
        Dtype::F8_E8M0 => {
            // ue8m0: the inverse rounded up to 2^e, and the forward factor its exact reciprocal.
            let e = (1.0 / forward).max(f32::MIN_POSITIVE).log2().ceil() as i32;
            let forward = exact::pow2(-e);
            return (vec![(e + 127) as u8], codes(&|v| v * forward));
        }
        Dtype::BF16 => BF16,
        _ => F16,
    };
    // compressed-tensors: s = amax/448 in the scale's dtype (its epsilon for a zero tile), then
    // each element x/s in that dtype.
    let round = |v: f32| (h.from)((h.to)(v));
    let s = match round(amax / MAX) {
        0.0 => h.eps,
        s => s,
    };
    ((h.to)(s).to_le_bytes().to_vec(), codes(&|v| round(v / s)))
}

/// The parts of `input.encoding` with the scale stored as `scale`, by that dtype's producer.
pub fn quantize_fp8_blockwise(input: &Input, scale: &Dtype) -> Result<Vec<OwnedPart>> {
    let fail = |d: String| Error::Encode(format!("fp8.blockwise: {d}"));
    let p = &input.encoding.params;
    let size = match p.get("block_size") {
        Some(Value::List(l)) if l.len() == 2 => {
            l.iter().map(Value::as_u64).collect::<Option<Vec<_>>>()
        }
        _ => None,
    };
    let Some([br, bc]) = size
        .as_deref()
        .map(|s| [s[0], s[1]])
        .filter(|s| s.iter().all(|n| *n > 0))
    else {
        return Err(fail("block_size is not [rows, cols]".into()));
    };
    let dtype = match scale {
        Dtype::F32 | Dtype::BF16 | Dtype::F16 | Dtype::F8_E8M0 => scale.clone(),
        _ => return Err(fail(format!("a {scale} scale is not F32, BF16, F16 or F8_E8M0"))),
    };
    let x = input.values()?;
    let shape = input.shape;
    let [.., n, c] = *shape else {
        return Err(fail(format!(
            "logical shape {shape:?} is not rank 2 or more"
        )));
    };
    let (n, c, br, bc) = (n as usize, c as usize, br as usize, bc as usize);
    let matrices = x.len().checked_div(n * c).unwrap_or(0);
    let (tr, tc) = (n.div_ceil(br), c.div_ceil(bc));
    let mut codes = vec![0u8; x.len()];
    let mut scale = Vec::new();
    let mut at = Vec::with_capacity(br * bc);
    for m in 0..matrices {
        for (ti, tj) in (0..tr).flat_map(|ti| (0..tc).map(move |tj| (ti, tj))) {
            at.clear();
            for i in ti * br..((ti + 1) * br).min(n) {
                let row = (m * n + i) * c;
                at.extend((tj * bc..((tj + 1) * bc).min(c)).map(|j| row + j));
            }
            let values: Vec<f32> = at.iter().map(|k| x[*k]).collect();
            let (s, q) = tile(&dtype, &values);
            scale.extend(s);
            for (k, q) in at.iter().zip(q) {
                codes[*k] = q;
            }
        }
    }
    let scale_shape = [&shape[..shape.len() - 2], &[tr as u64, tc as u64]].concat();
    Ok(vec![
        OwnedPart {
            role: "weight".into(),
            dtype: Dtype::F8_E4M3,
            shape: shape.to_vec(),
            bytes: codes,
        },
        OwnedPart {
            role: "scale".into(),
            dtype,
            shape: scale_shape,
            bytes: scale,
        },
    ])
}
