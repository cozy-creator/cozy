//! Checked decode plans prepared for TensorFS's prebuilt CUDA kernels. CUDA context, staging,
//! streams and launches belong to the read engine's shared driver.

use crate::format::decode::affine::{Codes as CodeType, Fill, Order};
use crate::format::decode::grid::{Grid, ScaleLayout};
use crate::format::decode::plan::{Bytes, Core, Elems, Plan, Post, Term};
use crate::format::decode::{round, Geometry};
use crate::format::{Dtype, Error, Result};

pub const FAMILIES: [u32; 6] = [80, 86, 89, 90, 100, 120];
pub const CORE_KINDS: [&str; 7] = [
    "codes",
    "unpack",
    "regular_h4",
    "rotated",
    "requantize",
    "grouped_rotated",
    "cast_metadata",
];

pub struct Binary {
    pub family: u32,
    pub kind: &'static str,
    pub dtype: &'static str,
    pub symbol: &'static str,
    pub shared: u32,
    pub warps: u32,
    pub image: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/kernels.rs"));

#[cfg(require_aot)]
const _: () = assert!(
    BINARIES.len()
        == FAMILIES.len() * (crate::format::decode::gguf::STRUCTS.len() + CORE_KINDS.len()) * 3
);

pub enum Prepared<'a> {
    Kernel(Box<Launch<'a>>),
    /// A scheme with a host-only plan: rounded bytes, ready for the engine's host source.
    Host(Vec<u8>),
}

/// Device arguments, in order: destination, words, slots, scale, term, table, global, post,
/// index, error (one zeroed i32), params. Factors are f32; words retain their stored bits.
/// `grouped_rotated`: scale=relative group scales, global=channel scales,
/// term=optional transposed correction, table=optional levels, post=unit factor.
pub struct Launch<'a> {
    pub kind: &'static str,
    pub dtype: Dtype,
    pub words: Bytes<'a>,
    pub slots: Vec<i32>,
    pub scale: Factors,
    pub term: Factors,
    pub table: Factors,
    pub global: Factors,
    pub post: Factors,
    pub index: Vec<u32>,
    pub params: [u64; 32],
}

/// Float kernel arguments are either normalized Host values or a typed external input.
/// A slot carries no pointer; the read engine binds it and casts metadata on the device.
pub enum Factors {
    Host(Vec<f32>),
    Slot {
        index: usize,
        dtype: Dtype,
        count: usize,
    },
}
impl From<Vec<f32>> for Factors {
    fn from(values: Vec<f32>) -> Self {
        Self::Host(values)
    }
}
impl TryFrom<Elems<'_>> for Factors {
    type Error = Error;
    fn try_from(values: Elems<'_>) -> Result<Self> {
        match values.bytes.binding() {
            Some((index, len)) => {
                let width = values
                    .dtype
                    .byte_len(&[1])
                    .ok_or_else(|| Error::Decode("unsupported factor dtype".into()))?
                    as usize;
                if width == 0 || !len.is_multiple_of(width) {
                    return Err(Error::Decode("factor byte length is not integral".into()));
                }
                Ok(Self::Slot {
                    index,
                    dtype: values.dtype,
                    count: len / width,
                })
            }
            None => Ok(Self::Host(values.floats()?)),
        }
    }
}
impl Factors {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(v) => v.len(),
            Self::Slot { count, .. } => *count,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn host(&self) -> Result<&[f32]> {
        match self {
            Self::Host(v) => Ok(v),
            Self::Slot { .. } => Err(Error::Decode(
                "resident factors require the I/O engine".into(),
            )),
        }
    }
}

impl Launch<'_> {
    pub fn binary(&self, family: u32) -> Option<&'static Binary> {
        self.stage_binary(family, self.kind)
    }

    /// A rotated plan first unpacks to i32, runs `regular_h4` (H₄ along one base-4 digit) once
    /// per stride 1, 4, … < B (`params[31]`), alternating two i32 buffers, which together apply
    /// `kron_regular_h4`; then runs `rotated` to scale and round.
    pub fn stage_binary(&self, family: u32, stage: &str) -> Option<&'static Binary> {
        BINARIES
            .iter()
            .find(|b| b.family == family && b.kind == stage && b.dtype == self.dtype.name())
    }

    pub fn blocks(&self) -> u64 {
        self.params[5].div_ceil(256)
    }
}

/// `count` logical elements beginning at `begin`. The plan and geometry come from `planned`;
/// this layer chooses a core, never a scheme. No driver or framework is touched here.
pub fn prepare<'a>(
    plan: Plan<'a>,
    g: Geometry,
    dtype: &Dtype,
    begin: u64,
    count: u64,
) -> Result<Prepared<'a>> {
    let total = g
        .rows
        .checked_mul(g.cols)
        .ok_or_else(|| Error::Decode("decode size overflow".into()))? as u64;
    if total > i64::MAX as u64 || begin.checked_add(count).is_none_or(|end| end > total) {
        return Err(Error::Decode(format!(
            "decode range {begin}+{count} exceeds {total}"
        )));
    }
    if !matches!(dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
        return Err(Error::Decode(format!(
            "kernel output dtype {dtype} is unsupported"
        )));
    }
    let mut launch = Launch {
        kind: "codes",
        dtype: dtype.clone(),
        words: (&[] as &[u8]).into(),
        slots: vec![0],
        scale: vec![1.0].into(),
        term: vec![0.0].into(),
        table: vec![0.0].into(),
        global: vec![1.0].into(),
        post: vec![1.0].into(),
        index: vec![0],
        params: [0; 32],
    };
    launch.params[..7].copy_from_slice(&[
        total,
        g.rows as u64,
        g.cols as u64,
        g.height as u64,
        begin,
        count,
        u64::from(g.swap),
    ]);
    match plan.post {
        Post::None => {}
        Post::One(f) => {
            launch.params[27] = 1;
            launch.post = f.try_into()?;
        }
        Post::Cols(f) => {
            launch.params[27] = 2;
            launch.post = f.try_into()?;
        }
    }
    let codes = match plan.core {
        Core::Codes(c) => c,
        Core::Rotated { codes, block } => {
            launch.kind = "rotated";
            launch.params[28] = block as u64;
            codes
        }
        Core::RequantizedRotation {
            codes,
            block,
            scale,
            correction,
        } => {
            let bits = codes.packing.bits();
            let group = codes.group;
            if group == 0
                || !g.cols.is_multiple_of(group)
                || !g.cols.is_multiple_of(4)
                || !(4..=65536).contains(&block)
                || !block.is_power_of_two()
                || block.trailing_zeros() % 2 != 0
                || !g.cols.is_multiple_of(block)
            {
                return Err(Error::Decode(
                    "invalid requantized rotation geometry".into(),
                ));
            }
            let width = codes.packing.row_bytes(g.cols);
            if width.checked_mul(g.rows) != Some(codes.bytes.len()) {
                return Err(Error::Decode(
                    "requantized codes have a wrong byte length".into(),
                ));
            }
            launch.kind = "grouped_rotated";
            launch.words = codes.bytes;
            launch.scale = codes.relative.try_into()?;
            launch.global = scale.try_into()?;
            let cells = g.rows.checked_mul(g.cols / group);
            if cells != Some(launch.scale.len()) || launch.global.len() != g.rows {
                return Err(Error::Decode(
                    "relative or channel scales have a wrong length".into(),
                ));
            }
            if let Some(correction) = correction {
                launch.term = correction.try_into()?;
                if cells != Some(launch.term.len()) {
                    return Err(Error::Decode("correction has a wrong length".into()));
                }
                launch.params[29] = 1;
            }
            if let Some(table) = codes.table {
                launch.table = table.try_into()?;
                if bits != 4 || launch.table.len() != 16 {
                    return Err(Error::Decode(
                        "requantized codebook is not16 four-bit levels".into(),
                    ));
                }
                launch.params[24] = 16;
            }
            launch.params[8] = bits as u64;
            launch.params[18] = group as u64;
            launch.params[28] = block as u64;
            launch.params[30] = 1 << (block.trailing_zeros() / 2);
            return Ok(Prepared::Kernel(Box::new(launch)));
        }
        Core::Structs { bytes, kind } => {
            launch.kind = kind.name;
            launch.words = bytes.into();
            launch.params[29] = kind.n as u64;
            launch.params[30] = kind.size as u64;
            return Ok(Prepared::Kernel(Box::new(launch)));
        }
        Core::Values(mut v) => {
            let post = launch.post.host()?;
            match launch.params[27] {
                1 => v.iter_mut().for_each(|x| *x *= post[0]),
                2 => v
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x *= post[i % g.cols]),
                _ => {}
            }
            if g.swap {
                v = crate::format::decode::swap_last_two(&v, g.height, g.cols);
            }
            return Ok(Prepared::Host(round(
                dtype,
                &v[begin as usize..(begin + count) as usize],
            )));
        }
    };
    let p = codes.packing;
    let (signed, offset) = match p.codes {
        CodeType::Signed => (1, 0),
        CodeType::Unsigned(n) => (0, n),
    };
    let (order, slots) = match p.order {
        Order::LowFirst => (0, vec![0]),
        Order::HighFirst => (1, vec![0]),
        Order::Slots(slots) => (
            2,
            (0..slots.len())
                .map(|i| slots.iter().position(|s| *s as usize == i).unwrap() as i32)
                .collect(),
        ),
    };
    let (rows, cols) = if codes.flat {
        (1, g.rows * g.cols)
    } else {
        (g.rows, g.cols)
    };
    let n = match (p.transposed, p.axis) {
        (false, 0) | (true, 1) => rows,
        _ => cols,
    };
    let fill = match p.fill {
        Fill::Word => 0,
        Fill::Dense => 1,
        Fill::Planar => 2,
    };
    launch.params[7..17].copy_from_slice(&[
        u64::from(codes.flat),
        p.bits as u64,
        p.word as u64,
        u64::from(p.transposed),
        p.axis as u64,
        fill,
        order,
        signed,
        offset as i64 as u64,
        p.words(n)? as u64,
    ]);
    launch.words = codes.bytes;
    launch.slots = slots;
    launch.scale = codes.scale.try_into()?;
    match codes.grid {
        Grid::Tensor => {}
        Grid::Row => launch.params[17] = 1,
        Grid::Col => launch.params[17] = 2,
        Grid::Blocks {
            size,
            layout,
            index,
        } => {
            launch.params[17] = 3;
            launch.params[18] = size as u64;
            launch.params[19] = match layout {
                ScaleLayout::RowMajor => 0,
                ScaleLayout::Transposed => 1,
                ScaleLayout::Cublas128x4 => 2,
            };
            if let Some(index) = index {
                launch.params[20] = 1;
                launch.index = index;
            }
        }
        Grid::Tiles { rows, cols } => {
            launch.params[17] = 4;
            launch.params[21] = rows as u64;
            launch.params[22] = cols as u64;
        }
    }
    match codes.term {
        Term::None => {}
        Term::Zero(v) => {
            launch.params[23] = 1;
            launch.term = v.try_into()?;
        }
        Term::Bias(v) => {
            launch.params[23] = 2;
            launch.term = v.try_into()?;
        }
    }
    if let Some(table) = codes.table {
        launch.params[24] = table.len() as u64;
        launch.table = table.into();
    }
    if let Some((v, divide)) = codes.global {
        launch.params[25] = if divide { 2 } else { 1 };
        launch.global = v.try_into()?;
        launch.params[26] = launch.global.len() as u64;
    }
    Ok(Prepared::Kernel(Box::new(launch)))
}
