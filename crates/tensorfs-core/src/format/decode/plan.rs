//! A scheme's decode as data (spec §6, building blocks): one core over the codes, with its scales,
//! zeros and factors as the producer stored them. The CPU runs a plan here; TensorFS's GPU kernels
//! run the same plan on a device (`crate::kernels`), so the two decode alike by construction.

use std::borrow::Cow;

use super::affine::{Packing, BYTES};
use super::grid::Grid;
use super::{blocks, elem, PartBytes};
use crate::format::{Dtype, Error, Result};

/// `core` over the `[R, C]` view, then each value `× post`.
pub struct Plan<'a> {
    pub core: Core<'a>,
    pub post: Post<'a>,
}

pub enum Core<'a> {
    Codes(Codes<'a>),
    /// Block structs (GGUF): `n` elements of a row in each struct of `size` bytes.
    Structs {
        bytes: &'a [u8],
        kind: &'static Struct,
    },
    /// The codes rotated, `y = q · H_B` per block of `B` columns in exact integers, then `y × s`.
    Rotated {
        codes: Codes<'a>,
        block: usize,
    },
    /// Values computed on the host: no GPU core serves the scheme.
    Values(Vec<f32>),
}

/// A block struct type: its name (the kernels' `STRUCTS`), elements, bytes and decode.
pub struct Struct {
    pub name: &'static str,
    pub n: usize,
    pub size: usize,
    pub decode: fn(&[u8], &mut [f32]),
}

/// A factor on every value: none, one, or one per column.
pub enum Post<'a> {
    None,
    One(Elems<'a>),
    Cols(Elems<'a>),
}

/// Elements as stored: a part's bytes in its dtype (`F32`, `BF16`, `F16`, an 8-bit format), or
/// bytes the plan derived (zeros as `U8`/`I8` codes).
pub struct Elems<'a> {
    pub dtype: Dtype,
    pub bytes: Cow<'a, [u8]>,
}

impl<'a> Elems<'a> {
    pub fn part(p: PartBytes<'a>) -> Elems<'a> {
        Elems {
            dtype: p.dtype.clone(),
            bytes: Cow::Borrowed(p.bytes),
        }
    }

    pub fn f32(v: f32) -> Elems<'static> {
        Elems {
            dtype: Dtype::F32,
            bytes: Cow::Owned(v.to_le_bytes().to_vec()),
        }
    }

    pub fn floats(&self) -> Vec<f32> {
        elem::floats(&self.dtype, &self.bytes)
    }
}

/// The term beside the scale, on the scale's grid: `(q − z) × s` or `q × s + b`.
pub enum Term<'a> {
    None,
    Zero(Elems<'a>),
    Bias(Elems<'a>),
}

/// `q × s`, `(q − z) × s`, `q × s + b` or `table[q] × s` on `grid`, then the global factor.
pub struct Codes<'a> {
    pub bytes: &'a [u8],
    pub packing: Packing<'a>,
    /// One lane over the whole tensor in row-major order (§5's `F4`), not one lane per row.
    pub flat: bool,
    /// `table[q]` in place of `q`; a code outside it is refused.
    pub table: Option<Vec<f32>>,
    pub scale: Elems<'a>,
    pub grid: Grid,
    pub term: Term<'a>,
    /// One factor, or one per matrix: `s ÷ g` first (`divide`), else `× g` last.
    pub global: Option<(Elems<'a>, bool)>,
}

impl<'a> Codes<'a> {
    /// `× s` of `scale` on `grid`, nothing else.
    pub fn new(
        bytes: &'a [u8],
        packing: Packing<'a>,
        scale: PartBytes<'a>,
        grid: Grid,
    ) -> Codes<'a> {
        Codes {
            bytes,
            packing,
            flat: false,
            table: None,
            scale: Elems::part(scale),
            grid,
            term: Term::None,
            global: None,
        }
    }

    /// An 8-bit element format's codes, each its table entry, `× s`.
    pub fn elements(weight: PartBytes<'a>, scale: PartBytes<'a>, grid: Grid) -> Codes<'a> {
        Codes {
            table: elem::table(weight.dtype),
            ..Codes::new(weight.bytes, BYTES, scale, grid)
        }
    }

    pub fn unpack(&self, rows: usize, cols: usize) -> Result<Vec<i32>> {
        match self.flat {
            true => self.packing.unpack(self.bytes, 1, rows * cols),
            false => self.packing.unpack(self.bytes, rows, cols),
        }
    }

    /// The values of codes `q`, each operation one f32 rounding, left to right.
    fn values(&self, q: &[i32], rows: usize, cols: usize, height: usize) -> Result<Vec<f32>> {
        let s = self.scale.floats();
        let (z, b) = match &self.term {
            Term::None => (None, None),
            Term::Zero(z) => (Some(z.floats()), None),
            Term::Bias(b) => (None, Some(b.floats())),
        };
        let g = self
            .global
            .as_ref()
            .map(|(g, divide)| (g.floats(), *divide));
        let mut out = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            let g = g
                .as_ref()
                .map(|(g, d)| (g[if g.len() == 1 { 0 } else { i / height }], *d));
            for j in 0..cols {
                let code = q[i * cols + j];
                let x = match &self.table {
                    None => code as f32,
                    Some(t) => *usize::try_from(code)
                        .ok()
                        .and_then(|k| t.get(k))
                        .ok_or_else(|| {
                            Error::Decode(format!("code {code} is outside a table of {}", t.len()))
                        })?,
                };
                let at = self.grid.at(rows, cols, height, i, j);
                let s = match g {
                    Some((g, true)) => s[at] / g,
                    _ => s[at],
                };
                let v = match (&z, &b) {
                    (Some(z), _) => (x - z[at]) * s,
                    (_, Some(b)) => x * s + b[at],
                    _ => x * s,
                };
                out.push(match g {
                    Some((g, false)) => v * g,
                    _ => v,
                });
            }
        }
        Ok(out)
    }
}

impl<'a> From<Codes<'a>> for Plan<'a> {
    fn from(codes: Codes<'a>) -> Plan<'a> {
        Plan {
            core: Core::Codes(codes),
            post: Post::None,
        }
    }
}

impl Plan<'_> {
    /// The `[R, C]` view's values, row-major; `height` rows to a matrix.
    pub fn values(self, rows: usize, cols: usize, height: usize) -> Result<Vec<f32>> {
        let mut v = match self.core {
            Core::Codes(c) => c.values(&c.unpack(rows, cols)?, rows, cols, height)?,
            Core::Rotated { codes, block } => {
                let mut q = codes.unpack(rows, cols)?;
                q.chunks_mut(block).for_each(hadamard);
                codes.values(&q, rows, cols, height)?
            }
            Core::Structs { bytes, kind } => {
                blocks::walk(rows, cols, bytes, kind.n, kind.size, |b, y| {
                    (kind.decode)(b, y);
                    Ok(())
                })?
            }
            Core::Values(v) => v,
        };
        match self.post {
            Post::None => {}
            Post::One(f) => {
                let f = f.floats()[0];
                v.iter_mut().for_each(|x| *x *= f);
            }
            Post::Cols(f) => {
                let f = f.floats();
                (v.iter_mut().enumerate()).for_each(|(k, x)| *x *= f[k % cols]);
            }
        }
        Ok(v)
    }
}

/// In place, `x · H` for H = H4 ⊗ … ⊗ H4 (unnormalized, entries ±1), x.len() a power of 4: each
/// quad `(a, b, c, d)` at a stride becomes `sum − 2·(d, c, b, a)`.
pub fn hadamard(x: &mut [i32]) {
    let mut stride = 1;
    while stride < x.len() {
        for base in (0..x.len()).filter(|i| (i / stride) % 4 == 0) {
            let quad = [0, 1, 2, 3].map(|j| x[base + j * stride]);
            let sum: i32 = quad.iter().sum();
            for (j, v) in quad.iter().rev().enumerate() {
                x[base + j * stride] = sum - 2 * v;
            }
        }
        stride *= 4;
    }
}
