//! `gptq.int2`, `gptq.int3`, `gptq.int4`, `gptq.int8` (spec §6 `gptq.*`): GPTQ's `b`-bit codes
//! packed along K in `I32` words, a zero and a scale per group of inputs, each input's group in
//! `g_idx`.

use super::affine::{Codes, Fill, Order, Packing};
use super::grid::{Grid, ScaleLayout};
use super::plan::{self, Elems, Term};
use super::{Allowed, Ctx, Dtypes, Param, Part, Plan, Scheme, Shape, View};
use crate::format::{Dtype, Result, Value};

pub const INT2: Scheme = scheme("gptq.int2");
pub const INT3: Scheme = scheme("gptq.int3");
pub const INT4: Scheme = scheme("gptq.int4");
pub const INT8: Scheme = scheme("gptq.int8");

pub const I32: Dtypes = Dtypes::Of(&["I32"]);
pub const HALF: Dtypes = Dtypes::Of(&["F16", "BF16"]);

/// `group_size`: a positive integer, or `-1` for one group per row.
pub const GROUP_SIZE: Param = Param {
    key: "group_size",
    allowed: Allowed::Check(
        |v| matches!(v, Value::Int(n) if *n == -1 || (1..=1 << 31).contains(n)),
        "a positive integer or -1",
    ),
};

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
    GROUP_SIZE,
    Param {
        key: "symmetric",
        allowed: Allowed::Bool,
    },
    Param {
        key: "act_order",
        allowed: Allowed::Bool,
    },
    Param {
        key: "zero_offset",
        allowed: Allowed::Uint(&[0, 1]),
    },
];

const PARTS: &[Part] = &[
    Part::req("qweight", I32, |c| {
        qweight(bits(c)).shape(&[], c.rows, c.cols)
    }),
    Part::req("qzeros", I32, |c| {
        qzeros(bits(c)).shape(&[groups(c) as u64], groups(c), c.rows)
    }),
    Part::req("scales", HALF, |c| blocks(c, None).shape(c)),
    Part::req("g_idx", I32, |c| Ok(Shape::Exact(vec![c.cols as u64]))),
];

/// The variant's code width: the digits after `int`.
pub fn bits(c: &Ctx) -> u32 {
    let digits = c.scheme.rsplit_once(".int").expect("an intB variant").1;
    digits.parse().expect("an intB variant")
}

/// The group size `g` along a row; `-1` is the whole row.
pub fn group(c: &Ctx) -> usize {
    match c.param("group_size") {
        Value::Int(-1) => c.cols.max(1),
        v => v.as_u64().expect("checked against the row") as usize,
    }
}

/// `W`: groups to a row.
pub fn groups(c: &Ctx) -> usize {
    c.cols.div_ceil(group(c))
}

fn blocks(c: &Ctx, index: Option<Vec<u32>>) -> Grid {
    Grid::Blocks {
        size: group(c),
        layout: ScaleLayout::Transposed,
        index,
    }
}

/// `qweight`: the view stored transposed (`[K, N]`), a dense stream along K.
pub fn qweight(bits: u32) -> Packing<'static> {
    Packing {
        bits,
        word: 32,
        transposed: true,
        axis: 0,
        fill: Fill::Dense,
        order: Order::LowFirst,
        codes: Codes::Unsigned(0),
    }
}

/// `qzeros`: `[W, N]` stored as is, a dense stream along N.
pub fn qzeros(bits: u32) -> Packing<'static> {
    Packing {
        axis: 1,
        transposed: false,
        ..qweight(bits)
    }
}

/// Little-endian `I32` elements.
pub fn ints(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect()
}

/// Zero codes as `U8` elements of a plan.
pub fn zeros(z: Vec<i32>) -> Elems<'static> {
    Elems {
        dtype: Dtype::U8,
        bytes: z.into_iter().map(|z| z as u8).collect::<Vec<u8>>().into(),
    }
}

/// `(q − z) × s(i, g_idx[j])`, `z` the stored zero plus `zero_offset`, modulo `2^b`.
fn decode<'a>(c: &Ctx<'a>) -> Result<Plan<'a>> {
    let (b, g, w) = (bits(c), group(c), groups(c));
    let mask = (1 << b) - 1;
    let offset = c.uint("zero_offset") as i32;
    let z: Vec<i32> = (qzeros(b)
        .unpack(c.part("qzeros").bytes, w, c.rows)?
        .into_iter())
    .map(|r| (r + offset) & mask)
    .collect();
    if c.flag("symmetric") && z.iter().any(|z| *z != 1 << (b - 1)) {
        return Err(c.fail(format!("symmetric, and a zero is not {}", 1 << (b - 1))));
    }
    let g_idx: Vec<u32> = ints(c.part("g_idx").bytes)
        .into_iter()
        .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
        .collect();
    if !c.flag("act_order") && (g_idx.iter().enumerate()).any(|(j, v)| *v as usize != j / g) {
        return Err(c.fail("act_order is false, and g_idx is not each column's ⌊j/g⌋"));
    }
    let grid = blocks(c, Some(g_idx));
    grid.check(c)?;
    let weight = c.part("qweight").bytes;
    Ok(plan::Codes {
        term: Term::Zero(zeros(z)),
        ..plan::Codes::new(weight, qweight(b), c.part("scales"), grid)
    }
    .into())
}
