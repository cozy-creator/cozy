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
