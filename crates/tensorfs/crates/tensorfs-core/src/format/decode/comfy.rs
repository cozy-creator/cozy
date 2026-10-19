//! ComfyUI's ConvRot integer formats (spec §6 `comfy.*`): comfy-kitchen's `AsymW4A8Int8Layout`
//! at 4 bits (`asym_w4a8_int8`) and 6 bits (`w6a8_int8`), and `TensorCoreConvRotW4A4Layout`
//! (`convrot_w4a4`). The codes are in the rotated basis; each block of `rotation_size` columns is
//! rotated back by `H_B`, as `int8.convrot` does.

use super::affine::{Codes, Fill, Order, Packing};
use super::grid::{Grid, ScaleLayout};
use super::int8::{hadamard, ROTATION_SIZE};
use super::{elem, Allowed, Ctx, Dtypes, Param, Part, Scheme, Shape, View};
use crate::format::{Result, Value};

const I8: Dtypes = Dtypes::Of(&["I8"]);
const F32: Dtypes = Dtypes::Of(&["F32"]);
const S_REL: Dtypes = Dtypes::Of(&["F8_E4M3", "F32"]);

pub const ASYM_W4A8_INT8: Scheme = Scheme {
    name: "comfy.asym_w4a8_int8",
    view: View::Matrix,
    params: &[
        Param {
            key: "group_size",
            allowed: Allowed::Check(w4_group, "4, 8 or a multiple of 16"),
        },
        ROTATION_SIZE,
    ],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 4, true)),
        Part::req("weight_s_rel", S_REL, |c| groups(c).shape(c)),
        Part::req("weight_s_channel", F32, |c| Grid::Row.shape(c)),
        Part::opt("weight_codebook", F32, |_| Ok(Shape::Count(16))),
        Part::opt(
            "weight_correction",
            Dtypes::Of(&["F32", "BF16", "F16"]),
            |c| correction(c).shape(c),
        ),
    ],
    decode: |c| grouped(c, 4),
};

pub const W6A8_INT8: Scheme = Scheme {
    name: "comfy.w6a8_int8",
    view: View::Matrix,
    params: &[
        Param {
            key: "group_size",
            allowed: Allowed::Check(w6_group, "a multiple of 16"),
        },
        ROTATION_SIZE,
    ],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 6, true)),
        Part::req("weight_s_rel", S_REL, |c| groups(c).shape(c)),
        Part::req("weight_s_channel", F32, |c| Grid::Row.shape(c)),
    ],
    decode: |c| grouped(c, 6),
};

pub const CONVROT_W4A4: Scheme = Scheme {
    name: "comfy.convrot_w4a4",
    view: View::Matrix,
    params: &[ROTATION_SIZE],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 4, false)),
        Part::req("weight_scale", F32, |c| Grid::Row.shape(c)),
    ],
    decode: |c| {
        let q = nibbles(Codes::Signed).unpack(c.part("weight").bytes, c.rows, c.cols)?;
        rotate(c, &q, &c.floats("weight_scale"), None)
    },
};

fn w4_group(v: &Value) -> bool {
    v.as_u64()
        .is_some_and(|g| g >= 4 && (16 % g == 0 || g % 16 == 0))
}

fn w6_group(v: &Value) -> bool {
    v.as_u64().is_some_and(|g| g >= 16 && g % 16 == 0)
}

/// `weight`: `bits` per element along the last axis, `[lead…, C·bits/8]`; `C` whole blocks of
/// `rotation_size` (and of `group_size`).
fn packed(c: &Ctx, bits: usize, group: bool) -> Result<Shape> {
    let b = c.uint("rotation_size");
    if !c.cols.is_multiple_of(b) || (group && !c.cols.is_multiple_of(c.uint("group_size"))) {
        let g = if group {
            format!(" and of {}", c.uint("group_size"))
        } else {
            String::new()
        };
        return Err(c.fail(format!("{} columns are not whole blocks of {b}{g}", c.cols)));
    }
    Ok(Shape::Exact(
        [c.lead(), &[(c.cols * bits / 8) as u64]].concat(),
    ))
}

/// `weight_s_rel`: row blocks of `group_size`, row-major.
fn groups(c: &Ctx) -> Grid<'static> {
    Grid::Blocks {
        size: c.uint("group_size"),
        layout: ScaleLayout::RowMajor,
        index: None,
    }
}

/// `weight_correction`: row blocks of `group_size`, transposed (`[C/g, R]`).
fn correction(c: &Ctx) -> Grid<'static> {
    Grid::Blocks {
        size: c.uint("group_size"),
        layout: ScaleLayout::Transposed,
        index: None,
    }
}

/// One `U8` lane of 4-bit codes, the even column low.
fn nibbles(codes: Codes) -> Packing<'static> {
    Packing {
        bits: 4,
        word: 8,
        transposed: false,
        axis: 1,
        fill: Fill::Word,
        order: Order::LowFirst,
        codes,
    }
}

/// A row's unsigned codes: `C/2` bytes of low nibbles, then (6-bit) `C/4` bytes of high pairs.
fn codes(row: &[u8], n: usize, bits: usize) -> Result<Vec<i32>> {
    let low = nibbles(Codes::Unsigned(0)).lane(&row[..n / 2], n)?;
    if bits == 4 {
        return Ok(low);
    }
    let pairs = Packing {
        bits: 2,
        ..nibbles(Codes::Unsigned(0))
    };
    let high = pairs.lane(&row[n / 2..], n)?;
    Ok(low.iter().zip(high).map(|(l, h)| l + 16 * h).collect())
}

/// `q8 = clamp(round(v × s_rel), ±127)` with `v` the code's level, then rotated back.
fn grouped(c: &Ctx, bits: usize) -> Result<Vec<f32>> {
    let grid = groups(c);
    let s_rel = c.floats("weight_s_rel");
    let table = c
        .has("weight_codebook")
        .then(|| c.floats("weight_codebook"));
    let mut q8 = Vec::with_capacity(c.rows * c.cols);
    let width = c.cols * bits / 8;
    for i in 0..c.rows {
        let row = &c.part("weight").bytes[i * width..(i + 1) * width];
        for (j, q) in codes(row, c.cols, bits)?.into_iter().enumerate() {
            let v = match &table {
                Some(t) => t[q as usize],
                None => (q - (1 << (bits - 1))) as f32,
            };
            let x = (v * s_rel[grid.at(c.rows, c.cols, i, j)]).round_ties_even();
            if x.is_nan() {
                return Err(c.fail(format!("element ({i}, {j}) scales to NaN")));
            }
            q8.push(x.clamp(-127.0, 127.0) as i32);
        }
    }
    let corr = c
        .has("weight_correction")
        .then(|| c.floats("weight_correction"));
    rotate(c, &q8, &c.floats("weight_s_channel"), corr.as_deref())
}

/// `((y × s[i]) + z) × 2^−k` per block of `B = 4^k` columns, `y = q · H_B` exact; `z` is the
/// correction's term, absent without one.
fn rotate(c: &Ctx, q: &[i32], s: &[f32], corr: Option<&[f32]>) -> Result<Vec<f32>> {
    let b = c.uint("rotation_size");
    let unit = elem::pow2(-(b.trailing_zeros() as i32 / 2));
    let per_row = c.cols / b;
    let terms = corr.map(|_| terms(c.uint("group_size"), b, per_row));
    let mut out = Vec::with_capacity(q.len());
    for (n, block) in q.chunks(b).enumerate() {
        let i = n / per_row;
        let mut y = block.to_vec();
        hadamard(&mut y);
        let z = match (corr, &terms) {
            (Some(corr), Some(terms)) => Some(term(corr, &terms[n % per_row], i, c.rows, b)),
            _ => None,
        };
        for (j, y) in y.iter().enumerate() {
            let v = *y as f32 * s[i];
            out.push(z.as_ref().map_or(v, |z| v + z[j]) * unit);
        }
    }
    Ok(out)
}

/// For each block of a row, the groups it meets in ascending order, each with
/// `M_b[j] = Σ_{j' in the group} H_B[j', j]`.
fn terms(g: usize, b: usize, blocks: usize) -> Vec<Vec<(usize, Vec<i32>)>> {
    (0..blocks)
        .map(|blk| {
            let (lo, hi) = (blk * b, (blk + 1) * b);
            (lo / g..hi.div_ceil(g))
                .map(|grp| {
                    let mut m: Vec<i32> = (lo..hi).map(|j| (j / g == grp) as i32).collect();
                    hadamard(&mut m);
                    (grp, m)
                })
                .collect()
        })
        .collect()
}

/// `z[j] = Σ_b c(i, b) × M_b[j]`, left to right; `c` stored `[C/g, R]`.
fn term(corr: &[f32], groups: &[(usize, Vec<i32>)], i: usize, rows: usize, b: usize) -> Vec<f32> {
    let mut z: Option<Vec<f32>> = None;
    for (grp, m) in groups {
        let c = corr[grp * rows + i];
        let t = m.iter().map(|m| c * *m as f32);
        z = Some(match z {
            None => t.collect(),
            Some(z) => z.iter().zip(t).map(|(a, t)| a + t).collect(),
        });
    }
    z.unwrap_or_else(|| vec![0.0; b])
}
