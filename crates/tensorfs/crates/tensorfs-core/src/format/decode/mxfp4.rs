//! `mxfp4` (spec §6 `mxfp4`): OCP MXFP4, an E2M1 element per weight and an `F8_E8M0` scale
//! per 32 elements of a row.

use super::grid::{Grid, ScaleLayout};
use super::{elem, logical, table, Allowed, Dtypes, Param, Part, Scheme, View};
use crate::format::Dtype;

const BLOCKS: Grid = Grid::Blocks {
    size: 32,
    layout: ScaleLayout::RowMajor,
    index: None,
};

pub const MXFP4: Scheme = Scheme {
    name: "mxfp4",
    view: View::Matrix,
    params: &[Param {
        key: "block_size",
        allowed: Allowed::Uint(&[32]),
    }],
    parts: &[
        Part::req("weight", Dtypes::Of(&["F4"]), logical),
        Part::req("scale", Dtypes::Of(&["F8_E8M0"]), |c| BLOCKS.shape(c)),
    ],
    decode: |c| {
        // §5's F4 order: element 2k in the low nibble of byte k, across the whole tensor.
        let w = c.part("weight").bytes;
        let codes: Vec<i32> = w
            .iter()
            .flat_map(|b| [b & 15, b >> 4].map(i32::from))
            .collect();
        let e2m1 = elem::table(&Dtype::F4).expect("F4 has a table");
        table::decode(&codes, &e2m1, &c.floats("scale"), &BLOCKS, c.rows, c.cols)
    },
};
