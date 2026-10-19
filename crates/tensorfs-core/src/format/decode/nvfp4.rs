//! `nvfp4` (spec §6 `nvfp4`): NVIDIA's NVFP4, an E2M1 element per weight, an E4M3 scale per 16
//! elements of a row and a per-tensor `scale_2`.

use super::affine::{Codes, Fill, Order, Packing};
use super::mxfp8::{blocks, SCALE_LAYOUT};
use super::{elem, logical, one, table, Allowed, Ctx, Dtypes, Param, Part, Scheme, View};
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
    ],
    parts: &[
        Part::req("weight", Dtypes::Of(&["F4"]), logical),
        Part::req("scale", Dtypes::Of(&["F8_E4M3"]), |c| {
            whole_blocks(c)?;
            blocks(c).shape(c)
        }),
        Part::req("scale_2", F32, one),
        Part::opt("input_scale", F32, one),
    ],
    decode,
};

fn whole_blocks(c: &Ctx) -> Result<()> {
    match c.cols.is_multiple_of(16) {
        true => Ok(()),
        false => Err(c.fail(format!("{} columns are not whole blocks of 16", c.cols))),
    }
}

/// `multiply`: `(table[q] × s) × scale_2`; `divide`: `table[q] × (s ÷ scale_2)`.
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
    let g = c.floats("scale_2")[0];
    let mut s = c.floats("scale");
    match c.text("global_scale") {
        "multiply" => {
            let out = table::decode(&q, &e2m1, &s, &blocks(c), c.rows, c.cols)?;
            Ok(out.into_iter().map(|v| v * g).collect())
        }
        _ => {
            s.iter_mut().for_each(|v| *v /= g);
            table::decode(&q, &e2m1, &s, &blocks(c), c.rows, c.cols)
        }
    }
}
