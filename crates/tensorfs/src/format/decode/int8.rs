//! `int8.tensorwise`, `int8.rowwise` and `int8.convrot` (spec §6).

use super::affine::SIGNED_BYTES;
use super::fp8::{axis_grid, per_axis, AXIS, SCALE, SCALE_DTYPE};
use super::grid::Grid;
use super::plan::{Codes, Core, Elems, Post};
use super::{elem, logical, one, Allowed, Ctx, Dtypes, Param, Part, Plan, Scheme, View};
use crate::format::Result;

const I8: Dtypes = Dtypes::Of(&["I8"]);
const F32: Dtypes = Dtypes::Of(&["F32"]);

/// Every stored ConvRot uses a power-of-four block, shared by all rotation schemes.
pub(super) const ROTATION_SIZE: Param = Param {
    key: "rotation_size",
    allowed: Allowed::Uint(&[4, 16, 64, 256, 1024, 4096, 16384, 65536]),
};

pub const TENSORWISE: Scheme = Scheme {
    name: "int8.tensorwise",
    view: View::Flat,
    params: &[SCALE_DTYPE],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", SCALE, one),
        Part::opt("input_scale", F32, one),
    ],
    decode: |c| Ok(scaled(c, Grid::Tensor).into()),
};

pub const ROWWISE: Scheme = Scheme {
    name: "int8.rowwise",
    view: View::Matrix,
    params: &[AXIS, SCALE_DTYPE],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", SCALE, per_axis),
    ],
    decode: |c| Ok(scaled(c, axis_grid(c)).into()),
};

pub const CONVROT: Scheme = Scheme {
    name: "int8.convrot",
    view: View::Matrix,
    params: &[
        Param {
            key: "rotation",
            allowed: Allowed::Text(&["hadamard"]),
        },
        ROTATION_SIZE,
        SCALE_DTYPE,
    ],
    parts: &[
        Part::req("weight", I8, logical),
        Part::req("scale", SCALE, |c| Grid::Row.shape(c)),
    ],
    decode: convrot,
};

/// `q × s(i, j)`.
fn scaled<'a>(c: &Ctx<'a>, grid: Grid) -> Codes<'a> {
    Codes::new(c.part("weight").bytes, SIGNED_BYTES, c.part("scale"), grid)
}

/// `(y × s[i]) × 2^−k` with `y` each group of `B = 4^k` times `H_B` (spec §6 `int8.convrot`).
fn convrot<'a>(c: &Ctx<'a>) -> Result<Plan<'a>> {
    let block = c.uint("rotation_size");
    if !c.cols.is_multiple_of(block) {
        return Err(c.fail(format!(
            "{} columns are not whole blocks of {block}",
            c.cols
        )));
    }
    let unit = elem::pow2(-(block.trailing_zeros() as i32 / 2));
    Ok(Plan {
        core: Core::Rotated {
            codes: scaled(c, Grid::Row),
            block,
        },
        post: Post::One(Elems::f32(unit)),
    })
}
