//! Checked decode plans prepared for TensorFS's prebuilt CUDA kernels. CUDA context, staging,
//! streams and launches belong to the read engine's shared driver.

use crate::format::decode::affine::{Codes as CodeType, Fill, Order};
use crate::format::decode::grid::{Grid, ScaleLayout};
use crate::format::decode::plan::{Core, Plan, Post, Term};
use crate::format::decode::{round, Geometry};
use crate::format::{Dtype, Error, Result};

pub const FAMILIES: [u32; 6] = [80, 86, 89, 90, 100, 120];

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

pub enum Prepared<'a> {
    Kernel(Box<Launch<'a>>),
    /// A scheme with a host-only plan: rounded bytes, ready for the engine's host source.
    Host(Vec<u8>),
}

/// Device arguments, in order: destination, words, slots, scale, term, table, global, post,
/// index, error (one zeroed i32), params. Factors are f32; words retain their stored bits.
pub struct Launch<'a> {
    pub kind: &'static str,
    pub dtype: Dtype,
    pub words: &'a [u8],
    pub slots: Vec<i32>,
    pub scale: Vec<f32>,
    pub term: Vec<f32>,
    pub table: Vec<f32>,
    pub global: Vec<f32>,
    pub post: Vec<f32>,
    pub index: Vec<u32>,
    pub params: [u64; 32],
}

impl Launch<'_> {
    pub fn binary(&self, family: u32) -> Option<&'static Binary> {
        self.stage_binary(family, self.kind)
    }

    /// A rotated plan first unpacks to i32, runs `hadamard` once per stride 1, 4, … < B
    /// (`params[31]`), alternating two i32 buffers, then runs `rotated` to scale and round.
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
    if begin.checked_add(count).is_none_or(|end| end > total) {
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
        words: &[],
        slots: vec![0],
        scale: vec![1.0],
        term: vec![0.0],
        table: vec![0.0],
        global: vec![1.0],
        post: vec![1.0],
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
            launch.post = f.floats();
        }
        Post::Cols(f) => {
            launch.params[27] = 2;
            launch.post = f.floats();
        }
    }
    let codes = match plan.core {
        Core::Codes(c) => c,
        Core::Rotated { codes, block } => {
            launch.kind = "rotated";
            launch.params[28] = block as u64;
            codes
        }
        Core::Structs { bytes, kind } => {
            launch.kind = kind.name;
            launch.words = bytes;
            launch.params[29] = kind.n as u64;
            launch.params[30] = kind.size as u64;
            return Ok(Prepared::Kernel(Box::new(launch)));
        }
        Core::Values(mut v) => {
            match launch.params[27] {
                1 => v.iter_mut().for_each(|x| *x *= launch.post[0]),
                2 => v
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x *= launch.post[i % g.cols]),
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
    launch.scale = codes.scale.floats();
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
            launch.term = v.floats();
        }
        Term::Bias(v) => {
            launch.params[23] = 2;
            launch.term = v.floats();
        }
    }
    if let Some(table) = codes.table {
        launch.params[24] = table.len() as u64;
        launch.table = table;
    }
    if let Some((v, divide)) = codes.global {
        launch.params[25] = if divide { 2 } else { 1 };
        launch.global = v.floats();
        launch.params[26] = launch.global.len() as u64;
    }
    Ok(Prepared::Kernel(Box::new(launch)))
}
