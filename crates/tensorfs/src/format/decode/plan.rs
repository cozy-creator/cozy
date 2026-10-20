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
    /// Integer codes reconstructed by relative scaling/rounding/saturation BEFORE the
    /// integer rotation; channel scaling and optional rotated correction follow it.
    RequantizedRotation {
        codes: RequantCodes<'a>,
        block: usize,
        scale: Elems<'a>,
        correction: Option<Elems<'a>>,
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

/// Storage for a group-scaled quantizer's unsigned integer codes.
#[derive(Clone, Copy)]
pub enum RequantPacking {
    Nibbles,
    /// Each row's nibble plane, then its high-two-bit plane (not a dense6-bit lane).
    SplitSix,
}

impl RequantPacking {
    pub fn bits(self) -> usize {
        match self {
            Self::Nibbles => 4,
            Self::SplitSix => 6,
        }
    }
}

/// `clamp(round_even(level[code] * relative[i,j/group]), -127,127)` as exact i32.
/// Without a table the level is `code - 2^(bits-1)`.
pub struct RequantCodes<'a> {
    pub bytes: &'a [u8],
    pub packing: RequantPacking,
    pub group: usize,
    pub relative: Elems<'a>,
    pub table: Option<Elems<'a>>,
}

impl RequantCodes<'_> {
    fn unpack(&self, rows: usize, cols: usize) -> Result<Vec<i32>> {
        let bits = self.packing.bits();
        if self.group == 0 || !cols.is_multiple_of(self.group) || !cols.is_multiple_of(4) {
            return Err(Error::Decode(
                "invalid requantized rotation geometry".into(),
            ));
        }
        let width = cols
            .checked_mul(bits)
            .and_then(|n| n.checked_div(8))
            .ok_or_else(|| Error::Decode("packed row size overflow".into()))?;
        if rows.checked_mul(width) != Some(self.bytes.len()) {
            return Err(Error::Decode(
                "requantized codes have a wrong byte length".into(),
            ));
        }
        let relative = self.relative.floats();
        if rows.checked_mul(cols / self.group) != Some(relative.len()) {
            return Err(Error::Decode("relative scales have a wrong length".into()));
        }
        let table = self.table.as_ref().map(Elems::floats);
        if table.as_ref().is_some_and(|t| bits != 4 || t.len() != 16) {
            return Err(Error::Decode(
                "requantized codebook is not16 levels of four-bit codes".into(),
            ));
        }
        let mut out = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            let row = &self.bytes[i * width..(i + 1) * width];
            for j in 0..cols {
                let lo = (row[j / 2] >> (4 * (j % 2))) & 15;
                let q = match self.packing {
                    RequantPacking::Nibbles => lo,
                    RequantPacking::SplitSix => {
                        lo | (((row[cols / 2 + j / 4] >> (2 * (j % 4))) & 3) << 4)
                    }
                };
                let level = table
                    .as_ref()
                    .map_or(q as f32 - (1 << (bits - 1)) as f32, |t| t[q as usize]);
                let x =
                    (level * relative[i * (cols / self.group) + j / self.group]).round_ties_even();
                if x.is_nan() {
                    return Err(Error::Decode(format!("element ({i}, {j}) scales to NaN")));
                }
                out.push(x.clamp(-127.0, 127.0) as i32);
            }
        }
        Ok(out)
    }
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
            Core::RequantizedRotation {
                codes,
                block,
                scale,
                correction,
            } => {
                if !(4..=65536).contains(&block)
                    || !block.is_power_of_two()
                    || block.trailing_zeros() % 2 != 0
                    || !cols.is_multiple_of(block)
                {
                    return Err(Error::Decode("invalid requantized rotation block".into()));
                }
                let group = codes.group;
                let mut q = codes.unpack(rows, cols)?;
                q.chunks_mut(block).for_each(hadamard);
                let scale = scale.floats();
                let correction = correction.as_ref().map(Elems::floats);
                if scale.len() != rows
                    || correction
                        .as_ref()
                        .is_some_and(|v| v.len() != rows * (cols / group))
                {
                    return Err(Error::Decode(
                        "channel scales or correction have a wrong length".into(),
                    ));
                }
                let mut out = Vec::with_capacity(q.len());
                for (e, y) in q.into_iter().enumerate() {
                    let (i, j) = (e / cols, e % cols);
                    let value = y as f32 * scale[i];
                    out.push(match &correction {
                        Some(c) => value + rotated_correction(c, rows, i, j, group, block),
                        None => value,
                    });
                }
                out
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

/// Sum of column `j` of H_B over its first `n` rows. H4[a,j] is -1 exactly
/// when a+j=3. A complete lower digit contributes2, so a base4 prefix costs log4(B),
/// without allocating a B-by-B matrix or expanding group correction to host values.
fn hadamard_prefix(n: usize, j: usize, block: usize) -> i32 {
    let mut weight = 1i32 << (block.trailing_zeros() / 2);
    if n == block {
        return weight;
    }
    let (mut left, mut span, mut sign, mut sum) = (n, block, 1, 0);
    while span > 1 {
        span /= 4;
        weight /= 2;
        let (a, b) = (left / span, (j / span) % 4);
        sum += sign * (a as i32 - 2 * i32::from(3 - b < a)) * weight;
        if a + b == 3 {
            sign = -sign;
        }
        left %= span;
    }
    sum
}

/// `sum_g correction[g,row] * sum_{t in group} H_B[t,j]`, ordered left to right.
fn rotated_correction(
    c: &[f32],
    rows: usize,
    row: usize,
    col: usize,
    group: usize,
    block: usize,
) -> f32 {
    let begin = col / block * block;
    let end = begin + block;
    let mut sum = None;
    for g in begin / group..end.div_ceil(group) {
        let lo = (g * group).max(begin) - begin;
        let hi = ((g + 1) * group).min(end) - begin;
        let m = hadamard_prefix(hi, col % block, block) - hadamard_prefix(lo, col % block, block);
        let term = c[g * rows + row] * m as f32;
        sum = Some(match sum {
            None => term,
            Some(sum) => sum + term,
        });
    }
    sum.unwrap_or(0.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_prefix_matches_independent_indicator_rotations() {
        for block in [4, 16, 64] {
            for n in 0..=block {
                let mut indicator: Vec<i32> = (0..block).map(|i| i32::from(i < n)).collect();
                hadamard(&mut indicator);
                for (j, expected) in indicator.into_iter().enumerate() {
                    assert_eq!(
                        hadamard_prefix(n, j, block),
                        expected,
                        "B{block} prefix{n} column{j}"
                    );
                }
            }
        }
    }
}
