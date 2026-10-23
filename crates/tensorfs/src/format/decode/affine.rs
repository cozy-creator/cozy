//! Bit packing (spec §6, building blocks): `b`-bit codes packed in words. The affine decode,
//! `(q − z) × s` on a scale grid, is a `plan::Codes`.

use super::Shape;
use crate::format::{Error, Result};

/// How codes fill a lane of words along the pack axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// `⌊w/b⌋` codes per word, in slots of `b` bits from bit 0; leftover bits are unused.
    Word,
    /// One little-endian bit stream: code `t` is bits `t·b … t·b + b − 1`, crossing words.
    Dense,
    /// Like `Word`, but code `t` of `n·⌊w/b⌋` is in word `t mod n` (the lane's halves, quarters,
    /// … share words: hqq, GGUF's q4_0).
    Planar,
}

/// Which slot of a word holds which of its codes (`Word` and `Planar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order<'a> {
    /// Code `e` in slot `e`.
    LowFirst,
    /// Code `e` in slot `⌊w/b⌋ − 1 − e`.
    HighFirst,
    /// Slot `s` holds code `slots[s]` (AWQ: `[0, 2, 4, 6, 1, 3, 5, 7]`).
    Slots(&'a [u8]),
}

/// How a slot's raw bits `r` become the code `q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codes {
    /// `b`-bit two's complement.
    Signed,
    /// `q = r − offset`.
    Unsigned(i32),
}

/// A packed part: the `[R, C]` view's codes, stored as `[R, C]` or transposed `[C, R]`, packed
/// along one axis of that stored array into words of `word` bits.
#[derive(Debug, Clone, Copy)]
pub struct Packing<'a> {
    /// 1 to 8.
    pub bits: u32,
    /// 8, 16 or 32: the part's container (`U8`, `I16`, `I32`, …), little-endian, read unsigned.
    pub word: u32,
    pub transposed: bool,
    /// The stored array's axis the codes are packed along: 0 or 1.
    pub axis: usize,
    pub fill: Fill,
    pub order: Order<'a>,
    pub codes: Codes,
}

/// One code per byte: unsigned (an 8-bit element format's code, a `U8` index), or `I8`.
pub const BYTES: Packing = Packing {
    bits: 8,
    word: 8,
    transposed: false,
    axis: 1,
    fill: Fill::Word,
    order: Order::LowFirst,
    codes: Codes::Unsigned(0),
};
pub const SIGNED_BYTES: Packing = Packing {
    codes: Codes::Signed,
    ..BYTES
};

fn bad(detail: String) -> Error {
    Error::Decode(detail)
}

impl Packing<'_> {
    fn per(&self) -> usize {
        (self.word / self.bits) as usize
    }

    /// Words in a lane of `n` codes.
    pub fn words(&self, n: usize) -> Result<usize> {
        match self.fill {
            Fill::Word => Ok(n.div_ceil(self.per())),
            Fill::Dense => Ok((n * self.bits as usize).div_ceil(self.word as usize)),
            Fill::Planar if n.is_multiple_of(self.per()) => Ok(n / self.per()),
            Fill::Planar => Err(bad(format!(
                "{n} codes do not fill planes of {}",
                self.per()
            ))),
        }
    }

    /// The stored array `[A, B]` of codes: the view, or its transpose.
    fn stored(&self, rows: usize, cols: usize) -> (usize, usize) {
        if self.transposed {
            (cols, rows)
        } else {
            (rows, cols)
        }
    }

    /// The packed part's shape: the stored array with its pack axis counted in words. A view
    /// stored as is and packed along its rows' last axis keeps its leading extents.
    pub fn shape(&self, lead: &[u64], rows: usize, cols: usize) -> Result<Shape> {
        let (a, b) = self.stored(rows, cols);
        Ok(Shape::Exact(match (self.transposed, self.axis) {
            (false, 1) => [lead, &[self.words(b)? as u64]].concat(),
            (_, 1) => vec![a as u64, self.words(b)? as u64],
            _ => vec![self.words(a)? as u64, b as u64],
        }))
    }

    /// For each code index within a word, its slot.
    fn slots(&self) -> Result<Vec<usize>> {
        let per = self.per();
        match (self.fill, self.order) {
            (_, Order::LowFirst) => Ok((0..per).collect()),
            (Fill::Dense, _) => Err(bad("a dense bit stream is low first".into())),
            (_, Order::HighFirst) => Ok((0..per).rev().collect()),
            (_, Order::Slots(s)) => {
                let mut slot = vec![usize::MAX; per];
                for (at, e) in s.iter().enumerate() {
                    if let Some(free) = slot.get_mut(*e as usize).filter(|v| **v == usize::MAX) {
                        *free = at;
                    }
                }
                if s.len() != per || slot.contains(&usize::MAX) {
                    return Err(bad(format!("slots {s:?} are not a permutation of {per}")));
                }
                Ok(slot)
            }
        }
    }

    /// Code `t` of a lane of `n` words, `word(k)` its `k`-th word.
    fn code(&self, slots: &[usize], n: usize, word: impl Fn(usize) -> u32, t: usize) -> i32 {
        let (b, w) = (self.bits as usize, self.word as usize);
        let mask = (1u64 << b) - 1;
        let raw = match self.fill {
            Fill::Dense => {
                let (k, off) = (t * b / w, t * b % w);
                let mut bits = word(k) as u64 >> off;
                if off + b > w {
                    bits |= (word(k + 1) as u64) << (w - off);
                }
                bits & mask
            }
            Fill::Word | Fill::Planar => {
                let (k, e) = match self.fill {
                    Fill::Word => (t / self.per(), t % self.per()),
                    _ => (t % n, t / n),
                };
                (word(k) as u64 >> (slots[e] * b)) & mask
            }
        } as i32;
        match self.codes {
            Codes::Signed => (raw << (32 - b)) >> (32 - b),
            Codes::Unsigned(offset) => raw - offset,
        }
    }

    /// The codes `q[i, j]` of the `[R, C]` view, row-major, from the packed bytes.
    pub fn unpack(&self, bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<i32>> {
        let slots = self.slots()?;
        let (a, b) = self.stored(rows, cols);
        let (n, along) = if self.axis == 0 { (a, b) } else { (b, a) };
        let nw = self.words(n)?;
        let wb = self.word as usize / 8;
        if bytes.len() != nw * along * wb {
            return Err(bad(format!(
                "{} bytes are not {nw}×{along} words",
                bytes.len()
            )));
        }
        let word = |at: usize| {
            let mut v = [0u8; 4];
            v[..wb].copy_from_slice(&bytes[at * wb..(at + 1) * wb]);
            u32::from_le_bytes(v)
        };
        let mut out = vec![0i32; rows * cols];
        for x in 0..a {
            for y in 0..b {
                let q = match self.axis {
                    0 => self.code(&slots, nw, |k| word(k * b + y), x),
                    _ => self.code(&slots, nw, |k| word(x * nw + k), y),
                };
                let (i, j) = if self.transposed { (y, x) } else { (x, y) };
                out[i * cols + j] = q;
            }
        }
        Ok(out)
    }

    /// The `n` codes of one lane: `bytes` is its words (a block struct's field, say).
    pub fn lane(&self, bytes: &[u8], n: usize) -> Result<Vec<i32>> {
        let slots = self.slots()?;
        let nw = self.words(n)?;
        let wb = self.word as usize / 8;
        if bytes.len() != nw * wb {
            return Err(bad(format!("{} bytes are not {nw} words", bytes.len())));
        }
        let word = |k: usize| {
            let mut v = [0u8; 4];
            v[..wb].copy_from_slice(&bytes[k * wb..(k + 1) * wb]);
            u32::from_le_bytes(v)
        };
        Ok((0..n).map(|t| self.code(&slots, nw, word, t)).collect())
    }
}
