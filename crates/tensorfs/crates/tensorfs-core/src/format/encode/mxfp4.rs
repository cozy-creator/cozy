//! `mxfp4`: OCP MXFP4 as gpt-oss writes it (triton_kernels' `downcast_to_mxfp_torch`, the
//! quantizer transformers runs), bit for bit: each block's scale is `amax/6` rounded up to a power
//! of two, so no element clips; elements round to the nearest E2M1 value, ties away from zero. A
//! row's last block may be short.

use super::{encoding, Encoder, Input, OwnedPart};
use crate::format::{Dtype, Encoding, Error, Result, Value};

pub const ENCODER: Encoder = Encoder {
    scheme: "mxfp4",
    quantize: Some(quantize),
    pack: None,
};

impl Encoding {
    pub fn mxfp4() -> Encoding {
        encoding("mxfp4", &[("block_size", 32u64.into())])
    }
}

fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let fail = |d: String| Error::Encode(format!("mxfp4: {d}"));
    if input
        .encoding
        .params
        .get("block_size")
        .and_then(Value::as_u64)
        != Some(32)
    {
        return Err(fail("block_size is not 32".into()));
    }
    let x = input.values()?;
    let shape = input.shape;
    let (Some(&c), true) = (shape.last(), shape.len() >= 2) else {
        return Err(fail(format!(
            "logical shape {shape:?} is not rank 2 or more"
        )));
    };
    if x.len() % 2 != 0 {
        return Err(fail(format!("{} elements do not fill F4 bytes", x.len())));
    }
    let mut codes = Vec::with_capacity(x.len());
    let mut scale = Vec::new();
    for row in x.chunks(c.max(1) as usize) {
        for block in row.chunks(32) {
            let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
            // The f32 bits rounded up to a power of two; zero stays zero (and quantizes to ±0).
            let up = ((amax / 6.0).to_bits() + 0x7f_ffff) & 0x7f80_0000;
            let inv = match up {
                0 => 0.0,
                _ => 1.0 / f32::from_bits(up),
            };
            scale.push((up >> 23) as u8);
            codes.extend(block.iter().map(|v| e2m1(v * inv)));
        }
    }
    let packed = codes.chunks(2).map(|p| p[0] | p[1] << 4).collect();
    let blocks = c.div_ceil(32);
    Ok(vec![
        OwnedPart {
            role: "weight".into(),
            dtype: Dtype::F4,
            shape: shape.to_vec(),
            bytes: packed,
        },
        OwnedPart {
            role: "scale".into(),
            dtype: Dtype::F8_E8M0,
            shape: [&shape[..shape.len() - 1], &[blocks]].concat(),
            bytes: scale,
        },
    ])
}

/// The E2M1 code nearest `v`, ties away from zero, saturating at 6; a negative `v` keeps its sign.
fn e2m1(v: f32) -> u8 {
    const UP: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    UP.iter().filter(|t| v.abs() >= **t).count() as u8 | if v.is_sign_negative() { 8 } else { 0 }
}
