//! `svdq.nvfp4` (spec §6 `svdq.nvfp4`): SVDQuant's weight, Nunchaku's svdq-fp4 in natural layout.
//! An nvfp4 residual of the smoothed weight, optionally a scale per row, and a 16-bit low-rank
//! branch.

use super::grid::Grid;
use super::nvfp4::{self, PARAMS};
use super::plan::{Elems, Post};
use super::{logical, one, Ctx, Dtypes, Part, Plan, Scheme, Shape, View};
use crate::format::Result;

const FACTORS: Dtypes = &["BF16", "F16", "F32"];

pub static SVDQ_NVFP4: Scheme = Scheme {
    name: "svdq.nvfp4",
    view: View::Matrix,
    params: PARAMS,
    parts: &[
        Part::req("weight", &["F4"], matrix),
        Part::req("scale", &["F8_E4M3"], nvfp4::scale),
        Part::req("scale_2", &["F32"], one),
        Part::opt("input_scale", &["F32"], one),
        Part::req("pre_quant_scale", &["F32", "BF16", "F16"], |c| Grid::Col.shape(c)),
        Part::opt("channel_scale", &["F32", "F8_E4M3"], |c| Grid::Row.shape(c)),
        Part::req("lowrank_down", FACTORS, |c| Ok(Shape::Exact(vec![rank(c)?, c.cols as u64]))),
        Part::req("lowrank_up", FACTORS, |c| Ok(Shape::Exact(vec![c.rows as u64, rank(c)?]))),
    ],
    decode,
};

/// The logical `[N, K]`: a stack is refused.
fn matrix(c: &Ctx) -> Result<Shape> {
    match c.shape.len() {
        2 => logical(c),
        _ => Err(c.fail(format!("logical shape {:?} is not [N, K]", c.shape))),
    }
}

/// `R`, the rank `lowrank_down` `[R, K]` states: at least 1.
fn rank(c: &Ctx) -> Result<u64> {
    match c.part("lowrank_down").shape {
        [r, _] if *r > 0 => Ok(*r),
        s => Err(c.fail(format!("part lowrank_down has shape {s:?}, not [R, {}], R ≥ 1", c.cols))),
    }
}

/// nvfp4's `(T[q] × s) × g` (or its `divide` form), `× c[i]` where there is a `channel_scale`,
/// `× p[j]`, then `+ up[i, r] × down[r, j]` for each `r` in order.
fn decode<'a>(c: &Ctx<'a>) -> Result<Plan<'a>> {
    let part = |role| Elems::part(c.part(role));
    Ok(Plan {
        core: nvfp4::decode(c)?.core,
        post: Post::LowRank {
            rows: c.has("channel_scale").then(|| part("channel_scale")),
            cols: part("pre_quant_scale"),
            down: part("lowrank_down"),
            up: part("lowrank_up"),
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::{bf16, elem};
    use crate::format::encode::{pack, Input};
    use crate::format::{decode, Dtype, Encoding, GlobalScale, NibbleOrder, PartBytes, ScaleLayout};

    /// A producer's arrays packed, then decoded at F32, are the law in f64 within f32 rounding:
    /// `T[q]·s·g·c[i]·p[j] + Σ_r up[i, r]·down[r, j]`.
    #[test]
    fn packed_parts_decode_to_the_law() {
        let (n, k, r) = (32, 64, 3);
        let wave = |i: usize, f: f64| (i as f64 * f).sin() as f32;
        let bits = |x: Vec<f32>| x.iter().flat_map(|x| bf16(*x).to_le_bytes()).collect::<Vec<u8>>();
        let le = |x: &[f32]| x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let codes: Vec<u8> = (0..n * k).map(|i| (i * 7 % 16) as u8).collect();
        let scale: Vec<u8> = (0..n * k / 16).map(|i| 0x30 + (i % 16) as u8).collect();
        let (g, p, c) = (
            [0.01f32],
            (0..k).map(|j| 0.5 + (j % 5) as f32 * 0.25).collect::<Vec<_>>(),
            (0..n).map(|i| 1.0 + wave(i, 0.3) / 2.0).collect::<Vec<_>>(),
        );
        let down = bits((0..r * k).map(|i| wave(i, 0.7) / 20.0).collect());
        let up = bits((0..n * r).map(|i| wave(i, 1.3) / 20.0).collect());
        let (g, p_bytes, c_bytes) = (le(&g), le(&p), le(&c));
        let (nk, n16, rk, nr) = ([n as u64, k as u64], [n as u64, k as u64 / 16], [r as u64, k as u64], [n as u64, r as u64]);
        let (k1, n1) = ([k as u64], [n as u64]);
        let held = [
            ("codes", Dtype::U8, &nk[..], &codes[..]),
            ("scale", Dtype::F8_E4M3, &n16, &scale),
            ("scale_2", Dtype::F32, &[], &g),
            ("pre_quant_scale", Dtype::F32, &k1, &p_bytes),
            ("channel_scale", Dtype::F32, &n1, &c_bytes),
            ("lowrank_down", Dtype::BF16, &rk, &down),
            ("lowrank_up", Dtype::BF16, &nr, &up),
        ];
        let arrays: BTreeMap<&str, PartBytes> = (held.iter())
        .map(|(role, dtype, shape, bytes)| (*role, PartBytes { dtype, shape, bytes }))
        .collect();
        let encoding = Encoding {
            scheme: "svdq.nvfp4".into(),
            ..Encoding::nvfp4(GlobalScale::Multiply, NibbleOrder::LowFirst, ScaleLayout::RowMajor)
        };
        let input = Input { dtype: &Dtype::F32, shape: &nk, encoding: &encoding, arrays: &arrays };
        let parts = pack(&input).unwrap();
        let roles: Vec<&str> = parts.iter().map(|p| p.role.as_str()).collect();
        assert_eq!(roles, ["weight", "scale", "scale_2", "pre_quant_scale", "channel_scale", "lowrank_down", "lowrank_up"]);
        let laid: BTreeMap<&str, PartBytes> = (parts.iter())
            .map(|p| (p.role.as_str(), PartBytes { dtype: &p.dtype, shape: &p.shape, bytes: &p.bytes }))
            .collect();
        let got = elem::floats(&Dtype::F32, &decode(&Dtype::F32, &nk, &encoding, &laid).unwrap());
        let (down, up) = (elem::floats(&Dtype::BF16, &down), elem::floats(&Dtype::BF16, &up));
        let law: Vec<f64> = (0..n * k)
            .map(|e| {
                let (i, j) = (e / k, e % k);
                let branch: f64 = (0..r).map(|t| up[i * r + t] as f64 * down[t * k + j] as f64).sum();
                elem::e2m1(codes[e]) as f64 * elem::e4m3(scale[e / 16]) as f64 * 0.01f32 as f64 * c[i] as f64 * p[j] as f64
                    + branch
            })
            .collect();
        let top = law.iter().fold(0f64, |a, x| a.max(x.abs()));
        let worst = got.iter().zip(&law).fold(0f64, |a, (x, y)| a.max((*x as f64 - y).abs()));
        assert!(worst <= 1e-6 * top, "{worst} of {top}");
        let stack = [1u64, n as u64, k as u64];
        let refused = pack(&Input { shape: &stack, ..input }).unwrap_err();
        assert!(refused.to_string().contains("[N, K]"), "{refused}");
    }
}
