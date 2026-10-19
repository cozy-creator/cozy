//! `bnb.nf4`, `bnb.fp4` and `bnb.int8` (spec §6): bitsandbytes' 4-bit tables over the flattened
//! tensor in blocks scaled by their absmax (itself 8-bit codes with `double_quant`), and LLM.int8()
//! rows.

use super::affine::{Codes, Fill, Order, Packing};
use super::grid::{Grid, ScaleLayout};
use super::{logical, one, table, Allowed, Ctx, Dtypes, Param, Part, Scheme, Shape, View};
use crate::format::Result;

/// Appendix A.3: NF4.
pub const NF4_MAP: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];
/// Appendix A.4: FP4 as bnb's kernels decode it; its stored `quant_map` holds code 8 as +0.
pub const FP4_MAP: [f32; 16] = [
    0.0,
    0.0052083335,
    0.6666667,
    1.0,
    0.33333334,
    0.5,
    0.16666667,
    0.25,
    -0.0,
    -0.0052083335,
    -0.6666667,
    -1.0,
    -0.33333334,
    -0.5,
    -0.16666667,
    -0.25,
];
/// Absmax blocks per nested block.
pub const NESTED: usize = 256;
/// The f32 nearest 1/127, which `bnb.int8` multiplies by.
pub const R127: u32 = 0x3c01_0204;

const F32: Dtypes = Dtypes::Of(&["F32"]);
const NIBBLES: Packing = Packing {
    bits: 4,
    word: 8,
    transposed: false,
    axis: 1,
    fill: Fill::Word,
    order: Order::HighFirst,
    codes: Codes::Unsigned(0),
};
const FOUR_BIT_PARAMS: &[Param] = &[
    Param {
        key: "block_size",
        allowed: Allowed::Uint(&[32, 64, 128, 256, 512, 1024, 2048, 4096]),
    },
    Param {
        key: "double_quant",
        allowed: Allowed::Bool,
    },
];
const FOUR_BIT_PARTS: &[Part] = &[
    Part::req("weight", Dtypes::Of(&["U8"]), |c| {
        exact(&[c.cols.div_ceil(2), 1])
    }),
    Part::req("absmax", Dtypes::Of(&["F32", "U8"]), |c| {
        exact(&[blocks(c)])
    }),
    Part::req("quant_map", F32, |_| exact(&[16])),
    Part::opt("nested_absmax", F32, |c| {
        exact(&[blocks(c).div_ceil(NESTED)])
    }),
    Part::opt("nested_quant_map", F32, |_| exact(&[256])),
    Part::opt("nested_offset", F32, one),
];

pub const NF4: Scheme = Scheme {
    name: "bnb.nf4",
    view: View::Flat,
    params: FOUR_BIT_PARAMS,
    parts: FOUR_BIT_PARTS,
    decode: |c| four_bit(c, &NF4_MAP),
};

pub const FP4: Scheme = Scheme {
    name: "bnb.fp4",
    view: View::Flat,
    params: FOUR_BIT_PARAMS,
    parts: FOUR_BIT_PARTS,
    decode: |c| four_bit(c, &FP4_MAP),
};

pub const INT8: Scheme = Scheme {
    name: "bnb.int8",
    view: View::Matrix,
    params: &[],
    parts: &[
        Part::req("weight", Dtypes::Of(&["I8"]), logical),
        Part::req("SCB", F32, |c| Grid::Row.shape(c)),
        Part::opt("weight_format", Dtypes::Of(&["U8"]), one),
    ],
    decode: int8,
};

fn exact(shape: &[usize]) -> Result<Shape> {
    Ok(Shape::Exact(shape.iter().map(|d| *d as u64).collect()))
}

fn blocks(c: &Ctx) -> usize {
    c.cols.div_ceil(c.uint("block_size"))
}

fn run(size: usize) -> Grid<'static> {
    Grid::Blocks {
        size,
        layout: ScaleLayout::RowMajor,
        index: None,
    }
}

/// `map[q] × s_b`, `s_b` the absmax of the code's block, or with `double_quant`
/// `(nested_quant_map[c_b] × nested_absmax[⌊b/256⌋]) + nested_offset`.
fn four_bit(c: &Ctx, map: &[f32; 16]) -> Result<Vec<f32>> {
    let dq = c.flag("double_quant");
    if c.part("absmax").dtype.name() != if dq { "U8" } else { "F32" } {
        return Err(c.fail("absmax is U8 codes exactly when double_quant"));
    }
    // As numbers: bnb stores FP4's code 8 as +0.
    if c.floats("quant_map") != map {
        return Err(c.fail("quant_map is not the scheme's table (Appendix A)"));
    }
    let nested = ["nested_absmax", "nested_quant_map", "nested_offset"];
    if nested.iter().any(|r| c.has(r) != dq) {
        return Err(c.fail("the nested parts are present exactly when double_quant"));
    }
    let absmax = if dq {
        let codes = table::bytes(c.part("absmax").bytes);
        let (nested, scale) = (c.floats("nested_quant_map"), c.floats("nested_absmax"));
        let offset = c.floats("nested_offset")[0];
        let s = table::decode(&codes, &nested, &scale, &run(NESTED), 1, blocks(c))?;
        s.into_iter().map(|v| v + offset).collect()
    } else {
        c.floats("absmax")
    };
    let q = NIBBLES.unpack(c.part("weight").bytes, 1, c.cols)?;
    table::decode(&q, map, &absmax, &run(c.uint("block_size")), 1, c.cols)
}

/// `(q × SCB[i]) × r`, `r` the f32 nearest 1/127: bitsandbytes multiplies by it.
fn int8(c: &Ctx) -> Result<Vec<f32>> {
    if c.shape.len() != 2 {
        return Err(c.fail(format!("shape {:?} is not [N, K]", c.shape)));
    }
    if c.has("weight_format") && c.part("weight_format").bytes != [0] {
        return Err(c.fail("weight_format is not 0 (row-major)"));
    }
    let (scb, r) = (c.floats("SCB"), f32::from_bits(R127));
    let w = c.part("weight").bytes;
    Ok((w.iter().enumerate())
        .map(|(k, q)| (*q as i8 as f32 * scb[k / c.cols]) * r)
        .collect())
}
