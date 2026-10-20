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

/// The word width of an integer container dtype.
pub fn word_bits(dtype: &crate::format::Dtype) -> Option<u32> {
    use crate::format::Dtype::*;
    match dtype {
        U8 | I8 => Some(8),
        U16 | I16 => Some(16),
        U32 | I32 => Some(32),
        _ => None,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::Dtype;
    use crate::format::decode::grid::{Grid, ScaleLayout};
    use crate::format::decode::plan::{self, Core, Elems, Plan, Post, Term};

    fn packing(bits: u32, word: u32, axis: usize, fill: Fill, order: Order) -> Packing {
        Packing {
            bits,
            word,
            transposed: false,
            axis,
            fill,
            order,
            codes: Codes::Unsigned(0),
        }
    }

    /// Packs codes the way the spec states, word by word, independently of `unpack`.
    fn words_le(words: &[u32], width: u32) -> Vec<u8> {
        let wb = width as usize / 8;
        words
            .iter()
            .flat_map(|w| w.to_le_bytes()[..wb].to_vec())
            .collect()
    }

    #[test]
    fn gptq_four_bit_low_first_along_k() {
        // GPTQ: qweight I32 [K/8, N], word (k/8, n) holds k%8 at bits 4·(k%8); stored [K, N].
        let (n, k) = (2, 16);
        let q = |row: usize, col: usize| ((row * 7 + col * 3) % 16) as u32;
        let mut words = vec![0u32; (k / 8) * n];
        for kk in 0..k {
            for nn in 0..n {
                words[(kk / 8) * n + nn] |= q(nn, kk) << (4 * (kk % 8));
            }
        }
        let p = Packing {
            transposed: true,
            ..packing(4, 32, 0, Fill::Word, Order::LowFirst)
        };
        assert_eq!(p.shape(&[], n, k).unwrap(), Shape::Exact(vec![2, 2]));
        let got = p.unpack(&words_le(&words, 32), n, k).unwrap();
        let want: Vec<i32> = (0..n)
            .flat_map(|r| (0..k).map(move |c| q(r, c) as i32))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn awq_order_along_n() {
        // AWQ: qweight I32 [K, N/8]; slot s of word (k, n/8) holds column 8·⌊n/8⌋ + order[s].
        const ORDER: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let (n, k) = (16, 3);
        let q = |row: usize, col: usize| ((row * 5 + col * 11 + 1) % 16) as u32;
        let mut words = vec![0u32; k * (n / 8)];
        for kk in 0..k {
            for w in 0..n / 8 {
                for (s, e) in ORDER.iter().enumerate() {
                    words[kk * (n / 8) + w] |= q(w * 8 + *e as usize, kk) << (4 * s);
                }
            }
        }
        let p = Packing {
            transposed: true,
            ..packing(4, 32, 1, Fill::Word, Order::Slots(&ORDER))
        };
        let got = p.unpack(&words_le(&words, 32), n, k).unwrap();
        let want: Vec<i32> = (0..n)
            .flat_map(|r| (0..k).map(move |c| q(r, c) as i32))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn three_bits_cross_words_densely() {
        // 32 three-bit codes in three u32 words: code 10 straddles words 0 and 1, code 21 words
        // 1 and 2. The stream is built bit by bit.
        let codes: Vec<u32> = (0..32).map(|t| (t * 5 + 3) % 8).collect();
        let mut words = [0u32; 3];
        for (t, c) in codes.iter().enumerate() {
            for bit in 0..3 {
                let at = t * 3 + bit;
                words[at / 32] |= ((c >> bit) & 1) << (at % 32);
            }
        }
        let p = packing(3, 32, 1, Fill::Dense, Order::LowFirst);
        assert_eq!(p.words(32).unwrap(), 3);
        let got = p.unpack(&words_le(&words, 32), 1, 32).unwrap();
        assert_eq!(got, codes.iter().map(|c| *c as i32).collect::<Vec<_>>());
        // compressed-tensors pads the last word: 11 codes in 2 words.
        assert_eq!(p.words(11).unwrap(), 2);
        let got = p.lane(&words_le(&words[..2], 32), 11).unwrap();
        assert_eq!(
            got,
            codes[..11].iter().map(|c| *c as i32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn words_leave_unused_bits_and_signs_extend() {
        // Word fill: ten 3-bit codes per u32, bits 30–31 unused (set here, ignored).
        let codes: Vec<i32> = (0..12).map(|t| (t % 8) - 4).collect();
        let mut words = [0xc000_0000u32, 0xc000_0000];
        for (t, c) in codes.iter().enumerate() {
            words[t / 10] |= ((*c as u32) & 7) << (3 * (t % 10));
        }
        let p = Packing {
            codes: Codes::Signed,
            ..packing(3, 32, 1, Fill::Word, Order::LowFirst)
        };
        assert_eq!(p.unpack(&words_le(&words, 32), 1, 12).unwrap(), codes);
        let offset = Packing {
            codes: Codes::Unsigned(4),
            ..p
        };
        let raw: Vec<i32> = codes.iter().map(|c| (*c as u32 & 7) as i32 - 4).collect();
        assert_eq!(offset.unpack(&words_le(&words, 32), 1, 12).unwrap(), raw);
    }

    #[test]
    fn nibbles_high_first_and_planar() {
        // bnb / ComfyUI: element 2i in the high nibble. hqq / GGUF q4_0: element t and t + n
        // share byte t (low, then high).
        let p = packing(4, 8, 1, Fill::Word, Order::HighFirst);
        assert_eq!(p.lane(&[0x12, 0xab], 4).unwrap(), [1, 2, 10, 11]);
        let planar = packing(4, 8, 1, Fill::Planar, Order::LowFirst);
        assert_eq!(planar.lane(&[0x12, 0xab], 4).unwrap(), [2, 11, 1, 10]);
        assert!(planar.lane(&[0x12], 1).is_err(), "planes are whole");
        // ModelOpt int4: rows 2r and 2r + 1 share byte (r, k), packed along axis 0.
        let rows = packing(4, 8, 0, Fill::Word, Order::LowFirst);
        assert_eq!(rows.shape(&[], 4, 2).unwrap(), Shape::Exact(vec![2, 2]));
        assert_eq!(
            rows.unpack(&[0x10, 0x32, 0x54, 0x76], 4, 2).unwrap(),
            [0, 2, 1, 3, 4, 6, 5, 7]
        );
    }

    #[test]
    fn affine_terms_on_a_grid() {
        // GPTQ act-order: g_idx picks each column's group; scales and zeros stored [W, R].
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (s, z) = (f32s(&[0.5, 2.0, 0.25, 4.0]), [8u8, 1, 7, 0]);
        let elems = |dtype, b: &[u8]| Elems {
            dtype,
            bytes: b.to_vec().into(),
        };
        let codes = |term, grid| plan::Codes {
            bytes: (&[0x53, 0x97, 0xf0, 0x18][..]).into(),
            packing: packing(4, 8, 1, Fill::Word, Order::LowFirst),
            flat: false,
            table: None,
            scale: elems(Dtype::F32, &s),
            grid,
            term,
            global: None,
        };
        let values = |c| {
            (Plan {
                core: Core::Codes(c),
                post: Post::None,
            })
            .values(2, 4, 2)
            .unwrap()
        };
        let g = Grid::Blocks {
            size: 2,
            layout: ScaleLayout::Transposed,
            index: Some(vec![1, 0, 1, 0]),
        };
        // q = [3, 5, 7, 9, 0, 15, 8, 1]; (i, j) → s[b·R + i] with b = g_idx[j].
        let got = values(codes(Term::Zero(elems(Dtype::U8, &z)), g));
        assert_eq!(got, [-1.0, -1.5, 0.0, 0.5, 0.0, 28.0, 32.0, 0.0]);
        let b = values(codes(Term::Bias(elems(Dtype::U8, &z)), Grid::Row));
        assert_eq!(b[5], 15.0 * 2.0 + 1.0);
        assert_eq!(values(codes(Term::None, Grid::Tensor))[1], 2.5);
    }
}
