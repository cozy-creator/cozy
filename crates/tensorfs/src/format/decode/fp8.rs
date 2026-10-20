//! `fp8.tensorwise`, `fp8.rowwise` and `fp8.columnwise` (spec §6): FP8 elements times one scale
//! for the tensor, or one per row, or one per column.

use super::grid::Grid;
use super::plan::Codes;
use super::{logical, one, Ctx, Dtypes, Part, Plan, Scheme, View};

const FP8: Dtypes = &["F8_E4M3", "F8_E5M2"];

pub const TENSORWISE: Scheme = Scheme {
    name: "fp8.tensorwise",
    view: View::Flat,
    params: &[],
    parts: &[
        Part::req("weight", FP8, logical),
        Part::req("scale", SCALE, one),
        Part::opt("input_scale", &["F32"], one),
    ],
    decode: |c| Ok(scaled(c, Grid::Tensor)),
};

pub const ROWWISE: Scheme = Scheme {
    name: "fp8.rowwise",
    view: View::Matrix,
    params: &[],
    parts: &[
        Part::req("weight", FP8, logical),
        Part::req("scale", SCALE, |c| Grid::Row.shape(c)),
    ],
    decode: |c| Ok(scaled(c, Grid::Row)),
};

pub const COLUMNWISE: Scheme = Scheme {
    name: "fp8.columnwise",
    view: View::Matrix,
    params: &[],
    parts: &[
        Part::req("weight", FP8, logical),
        Part::req("scale", SCALE, |c| Grid::Col.shape(c)),
    ],
    decode: |c| Ok(scaled(c, Grid::Col)),
};

/// A scale stored as the producer wrote it, its part's dtype read exactly.
pub const SCALE: Dtypes = &["F32", "BF16", "F16"];

/// `weight × s(i, j)`: the element table looked up by the weight's bytes.
fn scaled<'a>(c: &Ctx<'a>, grid: Grid) -> Plan<'a> {
    Codes::elements(c.part("weight"), c.part("scale"), grid).into()
}
