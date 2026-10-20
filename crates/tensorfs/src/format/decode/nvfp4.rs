//! `nvfp4` (spec §6 `nvfp4`): NVIDIA's NVFP4, an E2M1 element per weight, an E4M3 scale per 16
//! elements of a row, a global scale per matrix and optionally AWQ's per-column factor.

use super::affine::{Fill, Order, Packing, BYTES};
use super::grid::Grid;
use super::mxfp8::{blocks, SCALE_LAYOUT};
use super::plan::{Codes, Core, Elems, Post};
use super::{elem, logical, one, Allowed, Ctx, Dtypes, Param, Part, Plan, Scheme, Shape, View};
use crate::format::Result;

const F32: Dtypes = &["F32"];

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
    ],
    parts: &[
        Part::req("weight", &["F4"], logical),
        Part::req("scale", &["F8_E4M3"], |c| {
            whole_blocks(c)?;
            blocks(c).shape(c)
        }),
        Part::req("scale_2", F32, |c| per_matrix(c, "scale_2")),
        Part::opt("input_scale", F32, |c| per_matrix(c, "input_scale")),
        Part::opt("pre_quant_scale", &["F32", "BF16", "F16"], |c| Grid::Col.shape(c)),
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
fn decode<'a>(c: &Ctx<'a>) -> Result<Plan<'a>> {
    let order = match c.text("nibble_order") {
        "low_first" => Order::LowFirst,
        _ => Order::HighFirst,
    };
    let packing = Packing {
        bits: 4,
        fill: Fill::Word,
        order,
        ..BYTES
    };
    let global = (
        Elems::part(c.part("scale_2")),
        c.text("global_scale") == "divide",
    );
    let codes = Codes {
        table: Some((0..16).map(elem::e2m1).collect()),
        global: Some(global),
        ..Codes::new(c.part("weight").bytes, packing, c.part("scale"), blocks(c))
    };
    let post = match c.has("pre_quant_scale") {
        true => Post::Cols(Elems::part(c.part("pre_quant_scale"))),
        false => Post::None,
    };
    Ok(Plan {
        core: Core::Codes(codes),
        post,
    })
}
