//! `fp8.tensorwise` and `fp8.rowwise` (spec §6 `fp8.tensorwise`, `fp8.rowwise`): FP8 elements times one scale for the
//! tensor, or one per row or column.

use super::grid::Grid;
use super::plan::Codes;
use super::{logical, one, Allowed, Ctx, Dtypes, Param, Part, Plan, Scheme, Shape, View};
use crate::format::Result;

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
    params: &[AXIS],
    parts: &[
        Part::req("weight", FP8, logical),
        Part::req("scale", SCALE, per_axis),
    ],
    decode: |c| Ok(scaled(c, axis_grid(c))),
};

/// A scale stored as the producer wrote it, its part's dtype read exactly.
pub const SCALE: Dtypes = &["F32", "BF16", "F16"];

/// `axis` 0: a scale per row; 1: per column.
pub const AXIS: Param = Param {
    key: "axis",
    allowed: Allowed::Uint(&[0, 1]),
};

pub fn axis_grid(c: &Ctx) -> Grid {
    if c.uint("axis") == 0 {
        Grid::Row
    } else {
        Grid::Col
    }
}

/// One scale per row (`axis: 0`) or column (`axis: 1`), any shape.
pub fn per_axis(c: &Ctx) -> Result<Shape> {
    axis_grid(c).shape(c)
}

/// `weight × s(i, j)`: the element table looked up by the weight's bytes.
fn scaled<'a>(c: &Ctx<'a>, grid: Grid) -> Plan<'a> {
    Codes::elements(c.part("weight"), c.part("scale"), grid).into()
}
