//! `fp8.blockwise` (spec §6 `fp8.blockwise`): FP8 elements times one scale per tile of
//! `block_size = [rows, cols]` of each matrix (DeepSeek's `weight_scale_inv`, compressed-tensors'
//! block strategy).

use super::grid::Grid;
use super::{elem, logical, table, Allowed, Ctx, Dtypes, Param, Part, Scheme, View};
use crate::format::Value;

pub const BLOCKWISE: Scheme = Scheme {
    name: "fp8.blockwise",
    view: View::Matrix,
    params: &[
        Param {
            key: "block_size",
            allowed: Allowed::Check(block_size, "[rows, cols], each a positive integer"),
        },
        Param {
            key: "scale_dtype",
            allowed: Allowed::Text(&["F32", "BF16", "F16", "F8_E8M0"]),
        },
    ],
    parts: &[
        Part::req("weight", Dtypes::Of(&["F8_E4M3"]), logical),
        Part::req("scale", Dtypes::Param("scale_dtype"), |c| tiles(c).shape(c)),
    ],
    decode: |c| {
        let w = c.part("weight");
        let t = elem::table(w.dtype).expect("an 8-bit element format");
        let codes = table::bytes(w.bytes);
        table::decode(&codes, &t, &c.floats("scale"), &tiles(c), c.rows, c.cols)
    },
};

fn block_size(v: &Value) -> bool {
    matches!(v, Value::List(l) if l.len() == 2 && l.iter().all(|n| n.as_u64().is_some_and(|n| n > 0)))
}

/// The tile grid of `block_size` over each matrix.
fn tiles(c: &Ctx) -> Grid<'static> {
    let Value::List(b) = c.param("block_size") else {
        unreachable!("checked against the row")
    };
    let n = |k: usize| b[k].as_u64().expect("checked against the row") as usize;
    Grid::Tiles {
        rows: n(0),
        cols: n(1),
        height: c.shape[c.shape.len() - 2] as usize,
    }
}
