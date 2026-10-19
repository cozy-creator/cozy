//! `int8.tensorwise`, `int8.rowwise` and `int8.convrot` (spec §6).

use super::affine::{dequant, Offset};
use super::fp8::{axis_grid, per_axis, AXIS};
use super::grid::Grid;
use super::{elem, logical, one, Allowed, Ctx, Dtypes, Param, Part, Scheme, View};
use crate::format::Result;

const I8: Dtypes = Dtypes::Of(&["I8"]);
const F32: Dtypes = Dtypes::Of(&["F32"]);

pub const TENSORWISE: Scheme = Scheme {
    name: "int8.tensorwise",
    view: View::Flat,
    params: &[Param {
        key: "scale_dtype",
        allowed: Allowed::Text(&["F32", "BF16", "F16"]),
    }],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", Dtypes::Param("scale_dtype"), one),
        Part::opt("input_scale", F32, one),
    ],
    decode: |c| scaled(c, &Grid::Tensor),
};

pub const ROWWISE: Scheme = Scheme {
    name: "int8.rowwise",
    view: View::Matrix,
    params: &[AXIS],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", F32, per_axis),
    ],
    decode: |c| scaled(c, &axis_grid(c)),
};

pub const CONVROT: Scheme = Scheme {
    name: "int8.convrot",
    view: View::Matrix,
    params: &[
        Param {
            key: "rotation",
            allowed: Allowed::Text(&["hadamard"]),
        },
        ROTATION_SIZE,
    ],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", F32, |c| Grid::Row.shape(c)),
    ],
    decode: convrot,
};

/// `rotation_size`: `B = 4^k`, `1 ≤ k ≤ 8`.
pub const ROTATION_SIZE: Param = Param {
    key: "rotation_size",
    allowed: Allowed::Uint(&[4, 16, 64, 256, 1024, 4096, 16384, 65536]),
};

/// `q × s(i, j)`.
fn scaled(c: &Ctx, grid: &Grid) -> Result<Vec<f32>> {
    let q: Vec<i32> = c
        .part("weight")
        .bytes
        .iter()
        .map(|b| *b as i8 as i32)
        .collect();
    Ok(dequant(
        &q,
        &c.floats("scale"),
        Offset::None,
        grid,
        c.rows,
        c.cols,
    ))
}

/// `(y × s[i]) × 2^−k` with `y` each group of `B = 4^k` times `H_B` (spec §6 `int8.convrot`).
fn convrot(c: &Ctx) -> Result<Vec<f32>> {
    let block = c.uint("rotation_size");
    if !c.cols.is_multiple_of(block) {
        return Err(c.fail(format!(
            "{} columns are not whole blocks of {block}",
            c.cols
        )));
    }
    let s = c.floats("scale");
    let unit = elem::pow2(-(block.trailing_zeros() as i32 / 2));
    let mut out = Vec::with_capacity(c.rows * c.cols);
    for (g, group) in c.part("weight").bytes.chunks(block).enumerate() {
        let row = g * block / c.cols;
        let mut h: Vec<i32> = group.iter().map(|b| *b as i8 as i32).collect();
        hadamard(&mut h);
        out.extend(h.iter().map(|v| *v as f32 * s[row] * unit));
    }
    Ok(out)
}

/// In place, `x · H` for H = H4 ⊗ … ⊗ H4 (unnormalized, entries ±1), x.len() a power of 4.
pub fn hadamard(x: &mut [i32]) {
    let mut stride = 1;
    while stride < x.len() {
        for base in (0..x.len()).filter(|i| (i / stride) % 4 == 0) {
            let [a, b, c, d] = [0, 1, 2, 3].map(|j| x[base + j * stride]);
            let out = [a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a];
            for (j, v) in out.into_iter().enumerate() {
                x[base + j * stride] = v;
            }
        }
        stride *= 4;
    }
}
