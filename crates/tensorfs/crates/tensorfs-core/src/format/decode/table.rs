//! The table decode (spec §6, building blocks): `table[q] × s`, a code looked up in a fixed or stored table
//! (an element format, NF4, a codebook part), then scaled on a grid.

use super::grid::Grid;
use crate::format::{Error, Result};

/// `table[q[i, j]] × s(i, j)` over the `[R, C]` view; a code outside the table is refused.
pub fn decode(
    q: &[i32],
    table: &[f32],
    s: &[f32],
    grid: &Grid,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let code = q[i * cols + j];
            let Some(v) = usize::try_from(code).ok().and_then(|k| table.get(k)) else {
                return Err(Error::Decode(format!(
                    "code {code} is outside a table of {}",
                    table.len()
                )));
            };
            out.push(v * s[grid.at(rows, cols, i, j)]);
        }
    }
    Ok(out)
}

/// A part's bytes as codes (an 8-bit element format, or `U8` indices).
pub fn bytes(b: &[u8]) -> Vec<i32> {
    b.iter().map(|v| *v as i32).collect()
}
