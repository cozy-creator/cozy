//! `mxfp8` (spec §6 `mxfp8`): OCP MXFP8, an `F8_E8M0` scale per 32 elements of a row.

use super::grid::{Grid, ScaleLayout};
use super::{elem, logical, table, Allowed, Ctx, Dtypes, Param, Part, Scheme, View};

pub const MXFP8: Scheme = Scheme {
    name: "mxfp8",
    view: View::Matrix,
    params: &[
        Param {
            key: "block_size",
            allowed: Allowed::Uint(&[32]),
        },
        SCALE_LAYOUT,
    ],
    parts: &[
        Part::req("weight", Dtypes::Of(&["F8_E4M3", "F8_E5M2"]), logical),
        Part::req("scale", Dtypes::Of(&["F8_E8M0"]), |c| blocks(c).shape(c)),
    ],
    decode: |c| {
        let w = c.part("weight");
        let t = elem::table(w.dtype).expect("an 8-bit element format");
        table::decode(
            &table::bytes(w.bytes),
            &t,
            &c.floats("scale"),
            &blocks(c),
            c.rows,
            c.cols,
        )
    },
};

/// `scale_layout`: row-major, or ComfyUI's cuBLAS tiles.
pub const SCALE_LAYOUT: Param = Param {
    key: "scale_layout",
    allowed: Allowed::Text(&["row_major", "cublas_128x4"]),
};

/// Row blocks of `block_size`, stored as `scale_layout` says.
pub fn blocks(c: &Ctx) -> Grid<'static> {
    Grid::Blocks {
        size: c.uint("block_size"),
        layout: ScaleLayout::parse(c.text("scale_layout")).expect("checked against the row"),
        index: None,
    }
}
