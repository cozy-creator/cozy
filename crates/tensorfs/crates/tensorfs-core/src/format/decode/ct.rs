//! `ct.int1` … `ct.int8` (spec §6 `ct.*`): compressed-tensors' `pack-quantized`, signed `b`-bit
//! codes offset by `2^(b−1)` and packed along K in `I32` words, a scale (and, asymmetric, a zero
//! point packed along N) per group of inputs.

use super::affine::{dequant, Codes, Fill, Offset, Order, Packing};
use super::gptq::{bits, ints, I32};
use super::grid::{Grid, ScaleLayout};
use super::{Allowed, Ctx, Dtypes, Param, Part, Scheme, Shape, View};
use crate::format::Result;

pub const INT1: Scheme = scheme("ct.int1");
pub const INT2: Scheme = scheme("ct.int2");
pub const INT3: Scheme = scheme("ct.int3");
pub const INT4: Scheme = scheme("ct.int4");
pub const INT5: Scheme = scheme("ct.int5");
pub const INT6: Scheme = scheme("ct.int6");
pub const INT7: Scheme = scheme("ct.int7");
pub const INT8: Scheme = scheme("ct.int8");

const fn scheme(name: &'static str) -> Scheme {
    Scheme {
        name,
        view: View::Matrix,
        params: PARAMS,
        parts: PARTS,
        decode,
    }
}

const PARAMS: &[Param] = &[
    Param {
        key: "group_size",
        allowed: Allowed::Range(1, 1 << 31),
    },
    Param {
        key: "strategy",
        allowed: Allowed::Text(&["group", "channel"]),
    },
    Param {
        key: "symmetric",
        allowed: Allowed::Bool,
    },
    Param {
        key: "packing",
        allowed: Allowed::Text(&["word", "dense"]),
    },
];

const PARTS: &[Part] = &[
    Part::req("weight_packed", I32, |c| {
        packing(c, 1).shape(c.lead(), c.rows, c.cols)
    }),
    Part::req("weight_scale", Dtypes::Of(&["F32", "BF16", "F16"]), |c| {
        blocks(c, None).shape(c)
    }),
    Part::opt("weight_zero_point", I32, |c| {
        let Shape::Exact(one) = packing(c, 0).shape(&[], height(c), groups(c))? else {
            unreachable!("a packed shape is exact")
        };
        Ok(Shape::Exact([matrices_lead(c), &one[..]].concat()))
    }),
    Part::req("weight_shape", Dtypes::Of(&["I32", "I64"]), |c| {
        Ok(Shape::Exact(vec![c.shape.len() as u64]))
    }),
    Part::opt("weight_g_idx", I32, |c| {
        Ok(Shape::Exact(vec![c.cols as u64]))
    }),
];

/// `N`: the rows of each matrix, the logical shape's second-last extent.
fn height(c: &Ctx) -> usize {
    c.shape[c.shape.len() - 2] as usize
}

/// The logical shape before its last two extents.
fn matrices_lead<'a>(c: &Ctx<'a>) -> &'a [u64] {
    &c.shape[..c.shape.len() - 2]
}

fn groups(c: &Ctx) -> usize {
    c.cols.div_ceil(c.uint("group_size"))
}

/// Codes `q + 2^(b−1)` along `axis` of a `[R, C]` array stored as is.
fn packing(c: &Ctx, axis: usize) -> Packing<'static> {
    let b = bits(c);
    Packing {
        bits: b,
        word: 32,
        transposed: false,
        axis,
        fill: match c.text("packing") {
            "word" => Fill::Word,
            _ => Fill::Dense,
        },
        order: Order::LowFirst,
        codes: Codes::Unsigned(1 << (b - 1)),
    }
}

fn blocks<'a>(c: &Ctx, index: Option<&'a [u32]>) -> Grid<'a> {
    Grid::Blocks {
        size: c.uint("group_size"),
        layout: ScaleLayout::RowMajor,
        index,
    }
}

/// `(q − zp) × s(i, b)`, or `q × s(i, b)` when symmetric; `b` is `⌊j/g⌋`, or `weight_g_idx[j]`.
fn decode(c: &Ctx) -> Result<Vec<f32>> {
    let stored = c.part("weight_shape");
    let stated: Vec<i64> = match stored.dtype.name() {
        "I32" => ints(stored.bytes).into_iter().map(i64::from).collect(),
        _ => (stored.bytes.as_chunks::<8>().0.iter())
            .map(|v| i64::from_le_bytes(*v))
            .collect(),
    };
    if !stated.iter().zip(c.shape).all(|(a, b)| u64::try_from(*a) == Ok(*b)) {
        return Err(c.fail(format!("weight_shape {stated:?} is not {:?}", c.shape)));
    }
    let (g, w) = (c.uint("group_size"), groups(c));
    if c.text("strategy") == "channel" && g != c.cols {
        return Err(c.fail(format!("channel, and group_size {g} is not {}", c.cols)));
    }
    if c.flag("symmetric") == c.has("weight_zero_point") {
        return Err(c.fail("weight_zero_point is present exactly when not symmetric"));
    }
    let q = packing(c, 1).unpack(c.part("weight_packed").bytes, c.rows, c.cols)?;
    let mut z = Vec::new();
    if c.has("weight_zero_point") {
        // Each matrix's `[N, W]` zero points, packed along N.
        let (p, n) = (packing(c, 0), height(c));
        let matrix = p.words(n)? * w * 4;
        for m in c.part("weight_zero_point").bytes.chunks(matrix.max(1)) {
            z.extend(p.unpack(m, n, w)?.into_iter().map(|v| v as f32));
        }
    }
    let index = match c.has("weight_g_idx") {
        true => act_order(c, g, w)?,
        false => None,
    };
    let grid = blocks(c, index.as_deref());
    grid.check(c)?;
    let s = c.floats("weight_scale");
    let offset = match c.has("weight_zero_point") {
        true => Offset::Zero(&z),
        false => Offset::None,
    };
    Ok(dequant(&q, &s, offset, &grid, c.rows, c.cols))
}

/// `weight_g_idx` as each column's group. compressed-tensors groups the columns sorted by it in
/// runs of `g`, which is the stated group only when every group holds its `g` columns (the last
/// `C − (W − 1)·g`); a `−1` anywhere is its unset index, the column order.
fn act_order(c: &Ctx, g: usize, w: usize) -> Result<Option<Vec<u32>>> {
    let idx = ints(c.part("weight_g_idx").bytes);
    if idx.contains(&-1) {
        return Ok(None);
    }
    let mut count = vec![0usize; w];
    for v in &idx {
        match usize::try_from(*v).ok().and_then(|v| count.get_mut(v)) {
            Some(n) => *n += 1,
            None => return Err(c.fail(format!("weight_g_idx {v} is not one of {w} groups"))),
        }
    }
    let full = |b: usize| if b + 1 < w { g } else { c.cols - (w - 1) * g };
    if let Some(b) = (0..w).find(|b| count[*b] != full(*b)) {
        return Err(c.fail(format!("group {b} holds {} columns, not {}", count[b], full(b))));
    }
    Ok(Some(idx.into_iter().map(|v| v as u32).collect()))
}
