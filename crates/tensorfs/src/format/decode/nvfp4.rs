//! `nvfp4` (spec §6 `nvfp4`): NVIDIA's NVFP4, an E2M1 element per weight, an E4M3 scale per 16
//! elements of a row, a global scale per matrix and optionally AWQ's per-column factor.

use super::affine::{Codes, Fill, Order, Packing};
use super::grid::Grid;
use super::mxfp8::{SCALE_LAYOUT, blocks};
use super::{Allowed, Ctx, Dtypes, Param, Part, Scheme, Shape, View, elem, logical, one, table};
use crate::format::Result;

const F32: Dtypes = Dtypes::Of(&["F32"]);

pub const NVFP4: Scheme = Scheme {
    name: "nvfp4",
    view: View::Matrix,
    params: &[
        Param {
            key: "block_size",
            allowed: Allowed::Uint(&[16]),
        },
        Param {
            key: "global_scale",
            allowed: Allowed::Text(&["multiply", "divide"]),
        },
        Param {
            key: "nibble_order",
            allowed: Allowed::Text(&["low_first", "high_first"]),
        },
        SCALE_LAYOUT,
        Param {
            key: "pre_quant_scale_dtype",
            allowed: Allowed::PartDtype("pre_quant_scale", &["F32", "BF16", "F16"]),
        },
    ],
    parts: &[
        Part::req("weight", Dtypes::Of(&["F4"]), logical),
        Part::req("scale", Dtypes::Of(&["F8_E4M3"]), |c| {
            whole_blocks(c)?;
            blocks(c).shape(c)
        }),
        Part::req("scale_2", F32, |c| per_matrix(c, "scale_2")),
        Part::opt("input_scale", F32, |c| per_matrix(c, "input_scale")),
        Part::opt(
            "pre_quant_scale",
            Dtypes::Param("pre_quant_scale_dtype"),
            |c| Grid::Col.shape(c),
        ),
    ],
    decode,
};

fn whole_blocks(c: &Ctx) -> Result<()> {
    match c.cols.is_multiple_of(16) {
        true => Ok(()),
        false => Err(c.fail(format!("{} columns are not whole blocks of 16", c.cols))),
    }
}

/// One element, or `M`, one per matrix of `[…stack, N, C]`: any shape.
fn per_matrix(c: &Ctx, role: &str) -> Result<Shape> {
    match c.part(role).shape.iter().product::<u64>() {
        1 => one(c),
        _ => Ok(Shape::Count(c.shape[..c.shape.len() - 2].iter().product())),
    }
}

/// Matrix by matrix with its own `g`: `multiply` `((table[q] × s) × g) × p`; `divide`
/// `(table[q] × (s ÷ g)) × p`, `p` the `pre_quant_scale` of the column where there is one.
fn decode(c: &Ctx) -> Result<Vec<f32>> {
    let order = match c.text("nibble_order") {
        "low_first" => Order::LowFirst,
        _ => Order::HighFirst,
    };
    let packing = Packing {
        bits: 4,
        word: 8,
        transposed: false,
        axis: 1,
        fill: Fill::Word,
        order,
        codes: Codes::Unsigned(0),
    };
    let q = packing.unpack(c.part("weight").bytes, c.rows, c.cols)?;
    let e2m1: Vec<f32> = (0..16).map(elem::e2m1).collect();
    let (s, g) = (c.floats("scale"), c.floats("scale_2"));
    let divide = c.text("global_scale") == "divide";
    let (n, matrices) = table::matrices(c, &q, &s);
    let mut out = Vec::with_capacity(c.rows * c.cols);
    for (m, (q, s)) in matrices.into_iter().enumerate() {
        let g = g[if g.len() == 1 { 0 } else { m }];
        if divide {
            let s: Vec<f32> = s.iter().map(|v| v / g).collect();
            out.extend(table::decode(q, &e2m1, &s, &blocks(c), n, c.cols)?);
        } else {
            let v = table::decode(q, &e2m1, s, &blocks(c), n, c.cols)?;
            out.extend(v.into_iter().map(|v| v * g));
        }
    }
    if c.has("pre_quant_scale") {
        let p = c.floats("pre_quant_scale");
        out.iter_mut()
            .enumerate()
            .for_each(|(k, v)| *v *= p[k % c.cols]);
    }
    Ok(out)
}
