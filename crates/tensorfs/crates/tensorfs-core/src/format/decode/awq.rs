//! `awq.int4` (spec §6 `awq.int4`): AutoAWQ's GEMM layout, 4-bit codes packed along N in the
//! order `[0, 2, 4, 6, 1, 3, 5, 7]`, a zero and a scale per group of inputs.

use super::affine::{dequant, Codes, Fill, Offset, Order, Packing};
use super::gptq::{group, groups, GROUP_SIZE, HALF, I32};
use super::grid::{Grid, ScaleLayout};
use super::{Ctx, Part, Scheme, View};
use crate::format::Result;

pub const INT4: Scheme = Scheme {
    name: "awq.int4",
    view: View::Matrix,
    params: &[GROUP_SIZE],
    parts: &[
        Part::req("qweight", I32, |c| CODES.shape(&[], c.rows, c.cols)),
        Part::req("qzeros", I32, |c| {
            ZEROS.shape(&[groups(c) as u64], groups(c), c.rows)
        }),
        Part::req("scales", HALF, |c| blocks(c).shape(c)),
    ],
    decode,
};

/// The slot each of a word's eight codes sits in.
pub const ORDER: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// `qweight`: the view stored transposed (`[K, N]`), eight codes to a word along N.
pub const CODES: Packing = Packing {
    bits: 4,
    word: 32,
    transposed: true,
    axis: 1,
    fill: Fill::Word,
    order: Order::Slots(&ORDER),
    codes: Codes::Unsigned(0),
};

/// `qzeros`: `[W, N]` stored as is, packed like `qweight`.
pub const ZEROS: Packing = Packing {
    transposed: false,
    ..CODES
};

fn blocks(c: &Ctx) -> Grid<'static> {
    Grid::Blocks {
        size: group(c),
        layout: ScaleLayout::Transposed,
        index: None,
    }
}

/// `(q − z) × s(i, ⌊j/g⌋)`.
fn decode(c: &Ctx) -> Result<Vec<f32>> {
    let q = CODES.unpack(c.part("qweight").bytes, c.rows, c.cols)?;
    let z: Vec<f32> = (ZEROS
        .unpack(c.part("qzeros").bytes, groups(c), c.rows)?
        .into_iter())
    .map(|v| v as f32)
    .collect();
    let s = c.floats("scales");
    Ok(dequant(
        &q,
        &s,
        Offset::Zero(&z),
        &blocks(c),
        c.rows,
        c.cols,
    ))
}
