//! `mxfp4` (spec §6 `mxfp4`): OCP MXFP4, an E2M1 element per weight and an `F8_E8M0` scale
//! per 32 elements of a row; `transposed` (the registry's) stores the last two axes swapped.

use super::affine::{Fill, Packing, BYTES};
use super::grid::{Grid, ScaleLayout};
use super::plan::Codes;
use super::{elem, logical, Allowed, Param, Part, Scheme, View};
use crate::format::Dtype;

const BLOCKS: Grid = Grid::Blocks {
    size: 32,
    layout: ScaleLayout::RowMajor,
    index: None,
};

pub const MXFP4: Scheme = Scheme {
    name: "mxfp4",
    view: View::Matrix,
    params: &[
        Param {
            key: "block_size",
            allowed: Allowed::Uint(&[32]),
        },
        Param {
            key: "transposed",
            allowed: Allowed::Bool,
        },
    ],
    parts: &[
        Part::req("weight", &["F4"], logical),
        Part::req("scale", &["F8_E8M0"], |c| BLOCKS.shape(c)),
    ],
    decode: |c| {
        // §5's F4 order: element 2k in the low nibble of byte k, across the whole tensor.
        let packing = Packing {
            bits: 4,
            fill: Fill::Word,
            ..BYTES
        };
        Ok(Codes {
            flat: true,
            table: elem::table(&Dtype::F4),
            ..Codes::new(c.part("weight").bytes, packing, c.part("scale"), BLOCKS)
        }
        .into())
    },
};
