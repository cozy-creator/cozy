//! Codes to packed words: the inverse of the decode's `Packing::unpack`, so a calibrated
//! producer's codes become its own bytes again.

use crate::format::decode::affine::{Codes, Fill, Order, Packing};
use crate::format::{Error, Result};

/// The packed bytes of the `[R, C]` codes `q` (row-major); unused bits are zero.
pub fn pack(p: &Packing, q: &[i32], rows: usize, cols: usize) -> Result<Vec<u8>> {
    let fail = |d: String| Error::Encode(d);
    let (b, w) = (p.bits as usize, p.word as usize);
    let per = w / b;
    let (sa, sb) = if p.transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let (n, along) = if p.axis == 0 { (sa, sb) } else { (sb, sa) };
    let nw = p.words(n)?;
    let slot = |e: usize| match p.order {
        Order::LowFirst => e,
        Order::HighFirst => per - 1 - e,
        Order::Slots(s) => s
            .iter()
            .position(|v| *v as usize == e)
            .expect("a permutation"),
    };
    if !matches!(p.order, Order::LowFirst) && p.fill == Fill::Dense {
        return Err(fail("a dense bit stream is low first".into()));
    }
    let mut words = vec![0u64; nw * along];
    for i in 0..rows {
        for j in 0..cols {
            let v = q[i * cols + j];
            let raw = match p.codes {
                Codes::Signed if v >= -(1 << (b - 1)) && v < 1 << (b - 1) => {
                    (v as u32) & ((1 << b) - 1)
                }
                Codes::Unsigned(o) if (0..1 << b).contains(&(v + o)) => (v + o) as u32,
                _ => return Err(fail(format!("code {v} does not fit {b} bits"))),
            } as u64;
            let (x, y) = if p.transposed { (j, i) } else { (i, j) };
            let (t, lane) = if p.axis == 0 { (x, y) } else { (y, x) };
            // (word within the lane, bit offset)
            let (k, off) = match p.fill {
                Fill::Dense => (t * b / w, t * b % w),
                Fill::Word => (t / per, slot(t % per) * b),
                Fill::Planar => (t % nw, slot(t / nw) * b),
            };
            let at = |k: usize| {
                if p.axis == 0 {
                    k * sb + lane
                } else {
                    lane * nw + k
                }
            };
            words[at(k)] |= raw << off;
            if off + b > w {
                words[at(k + 1)] |= raw >> (w - off);
            }
        }
    }
    let mask = (1u64 << w) - 1;
    Ok(words
        .iter()
        .flat_map(|v| ((v & mask) as u32).to_le_bytes()[..w / 8].to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_layout(bits: u32, word: u32) -> Vec<Packing<'static>> {
        const AWQ: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let mut out = Vec::new();
        for transposed in [false, true] {
            for axis in [0, 1] {
                for (fill, order) in [
                    (Fill::Word, Order::LowFirst),
                    (Fill::Word, Order::HighFirst),
                    (Fill::Planar, Order::HighFirst),
                    (Fill::Dense, Order::LowFirst),
                    (Fill::Word, Order::Slots(&AWQ)),
                ] {
                    if matches!(order, Order::Slots(_)) && word / bits != 8 {
                        continue;
                    }
                    for codes in [
                        Codes::Signed,
                        Codes::Unsigned(0),
                        Codes::Unsigned(1 << (bits - 1)),
                    ] {
                        out.push(Packing {
                            bits,
                            word,
                            transposed,
                            axis,
                            fill,
                            order,
                            codes,
                        });
                    }
                }
            }
        }
        out
    }

    #[test]
    fn pack_inverts_unpack() {
        for (bits, word) in [
            (1, 8),
            (2, 8),
            (3, 32),
            (4, 8),
            (4, 32),
            (5, 16),
            (6, 32),
            (7, 8),
            (8, 32),
        ] {
            for p in every_layout(bits, word) {
                let per = (word / bits) as usize;
                // Planar needs whole planes along the pack axis.
                let (rows, cols) = (2 * per, 3 * per);
                let lo = match p.codes {
                    Codes::Signed => -(1 << (bits - 1)),
                    Codes::Unsigned(o) => -o,
                };
                let q: Vec<i32> = (0..rows * cols)
                    .map(|t| lo + ((t * 7 + t / 3) % (1 << bits)) as i32)
                    .collect();
                let bytes = pack(&p, &q, rows, cols).unwrap();
                assert_eq!(p.unpack(&bytes, rows, cols).unwrap(), q, "{p:?}");
            }
        }
    }

    #[test]
    fn producers_bytes() {
        // GPTQ 4-bit, qweight I32 [K/8, N]: word (k/8, n), code k%8 at bits 4·(k%8).
        let p = Packing {
            bits: 4,
            word: 32,
            transposed: true,
            axis: 0,
            fill: Fill::Word,
            order: Order::LowFirst,
            codes: Codes::Unsigned(0),
        };
        let q: Vec<i32> = (0..8).collect(); // [N = 1, K = 8]
        assert_eq!(pack(&p, &q, 1, 8).unwrap(), 0x7654_3210u32.to_le_bytes());
        // AWQ: slot s holds column order[s].
        let awq = Packing {
            transposed: true,
            axis: 1,
            order: Order::Slots(&[0, 2, 4, 6, 1, 3, 5, 7]),
            ..p
        };
        assert_eq!(pack(&awq, &q, 8, 1).unwrap(), 0x7531_6420u32.to_le_bytes());
        // compressed-tensors 3-bit, dense along K.
        let ct = Packing {
            bits: 3,
            transposed: false,
            axis: 1,
            fill: Fill::Dense,
            ..p
        };
        let q: Vec<i32> = vec![7; 11];
        assert_eq!(
            pack(&ct, &q, 1, 11).unwrap(),
            [0xff, 0xff, 0xff, 0xff, 0x01, 0, 0, 0]
        );
        assert!(pack(&ct, &[8], 1, 1).is_err(), "8 does not fit 3 bits");
    }
}
