//! ComfyUI's ConvRot integer formats (spec §6 `comfy.*`): comfy-kitchen's `AsymW4A8Int8Layout`
//! at 4 bits (`asym_w4a8_int8`) and 6 bits (`w6a8_int8`), and `TensorCoreConvRotW4A4Layout`
//! (`convrot_w4a4`). The codes are in the rotated basis; each block of `rotation_size` columns is
//! rotated back by `H_B`, as `int8.convrot` does.

use super::affine::{Codes, Fill, Order, Packing};
use super::grid::{Grid, ScaleLayout};
use super::int8::ROTATION_SIZE;
use super::plan::{Codes as PlanCodes, Core, Elems, Plan, Post, RequantCodes, RequantPacking};
use super::{elem, Allowed, Ctx, Dtypes, Param, Part, Scheme, Shape, View};
use crate::format::{Result, Value};

const I8: Dtypes = Dtypes::Of(&["I8"]);
const F32: Dtypes = Dtypes::Of(&["F32"]);
const S_REL: Dtypes = Dtypes::Of(&["F8_E4M3", "F32"]);

pub const ASYM_W4A8_INT8: Scheme = Scheme {
    name: "comfy.asym_w4a8_int8",
    view: View::Matrix,
    params: &[
        Param {
            key: "group_size",
            allowed: Allowed::Check(w4_group, "4, 8 or a multiple of 16"),
        },
        ROTATION_SIZE,
    ],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 4, true)),
        Part::req("weight_s_rel", S_REL, |c| groups(c).shape(c)),
        Part::req("weight_s_channel", F32, |c| Grid::Row.shape(c)),
        Part::opt("weight_codebook", F32, |_| Ok(Shape::Count(16))),
        Part::opt(
            "weight_correction",
            Dtypes::Of(&["F32", "BF16", "F16"]),
            |c| correction(c).shape(c),
        ),
    ],
    decode: |c| grouped(c, 4),
};

pub const W6A8_INT8: Scheme = Scheme {
    name: "comfy.w6a8_int8",
    view: View::Matrix,
    params: &[
        Param {
            key: "group_size",
            allowed: Allowed::Check(w6_group, "a multiple of 16"),
        },
        ROTATION_SIZE,
    ],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 6, true)),
        Part::req("weight_s_rel", S_REL, |c| groups(c).shape(c)),
        Part::req("weight_s_channel", F32, |c| Grid::Row.shape(c)),
    ],
    decode: |c| grouped(c, 6),
};

pub const CONVROT_W4A4: Scheme = Scheme {
    name: "comfy.convrot_w4a4",
    view: View::Matrix,
    params: &[ROTATION_SIZE],
    parts: &[
        Part::req("weight", I8, |c| packed(c, 4, false)),
        Part::req("weight_scale", F32, |c| Grid::Row.shape(c)),
    ],
    decode: |c| {
        let codes = PlanCodes::new(
            c.part("weight").bytes,
            nibbles(Codes::Signed),
            c.part("weight_scale"),
            Grid::Row,
        );
        Ok(rotation(
            c,
            Core::Rotated {
                codes,
                block: c.uint("rotation_size"),
            },
        ))
    },
};

fn w4_group(v: &Value) -> bool {
    v.as_u64()
        .is_some_and(|g| g >= 4 && (16 % g == 0 || g % 16 == 0))
}

fn w6_group(v: &Value) -> bool {
    v.as_u64().is_some_and(|g| g >= 16 && g % 16 == 0)
}

/// `weight`: `bits` per element along the last axis, `[lead…, C·bits/8]`; `C` whole blocks of
/// `rotation_size` (and of `group_size`).
fn packed(c: &Ctx, bits: usize, group: bool) -> Result<Shape> {
    let b = c.uint("rotation_size");
    if !c.cols.is_multiple_of(b) || (group && !c.cols.is_multiple_of(c.uint("group_size"))) {
        let g = if group {
            format!(" and of {}", c.uint("group_size"))
        } else {
            String::new()
        };
        return Err(c.fail(format!("{} columns are not whole blocks of {b}{g}", c.cols)));
    }
    let width = match bits {
        4 => c.cols / 2,
        _ => (c.cols / 4) * 3,
    };
    Ok(Shape::Exact([c.lead(), &[width as u64]].concat()))
}

/// `weight_s_rel`: row blocks of `group_size`, row-major.
fn groups(c: &Ctx) -> Grid {
    Grid::Blocks {
        size: c.uint("group_size"),
        layout: ScaleLayout::RowMajor,
        index: None,
    }
}

/// `weight_correction`: row blocks of `group_size`, transposed (`[C/g, R]`).
fn correction(c: &Ctx) -> Grid {
    Grid::Blocks {
        size: c.uint("group_size"),
        layout: ScaleLayout::Transposed,
        index: None,
    }
}

/// One byte's nibbles, even column low, independently signed for W4A4.
fn nibbles(codes: Codes) -> Packing<'static> {
    Packing {
        bits: 4,
        word: 8,
        transposed: false,
        axis: 1,
        fill: Fill::Word,
        order: Order::LowFirst,
        codes,
    }
}

fn rotation<'a>(c: &Ctx<'a>, core: Core<'a>) -> Plan<'a> {
    let b = c.uint("rotation_size");
    Plan {
        core,
        post: Post::One(Elems::f32(elem::pow2(-(b.trailing_zeros() as i32 / 2)))),
    }
}

/// Reconstruct rounded/saturated int8 values before the integer rotation. The plan owns
/// only scale/table metadata; GPU lowering never expands the logical weight on the host.
fn grouped<'a>(c: &Ctx<'a>, bits: usize) -> Result<Plan<'a>> {
    let codes = RequantCodes {
        bytes: c.part("weight").bytes,
        packing: if bits == 4 {
            RequantPacking::Nibbles
        } else {
            RequantPacking::SplitSix
        },
        group: c.uint("group_size"),
        relative: Elems::part(c.part("weight_s_rel")),
        table: c
            .has("weight_codebook")
            .then(|| Elems::part(c.part("weight_codebook"))),
    };
    Ok(rotation(
        c,
        Core::RequantizedRotation {
            codes,
            block: c.uint("rotation_size"),
            scale: Elems::part(c.part("weight_s_channel")),
            correction: c
                .has("weight_correction")
                .then(|| Elems::part(c.part("weight_correction"))),
        },
    ))
}
