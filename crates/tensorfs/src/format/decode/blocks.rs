//! Block structs (spec §6, building blocks): a `U8 [lead…, C/n, size]` part of fixed-size structs, each
//! holding `n` consecutive elements of a row (GGUF's `block_q4_0` and kin). The walker hands
//! each struct to the scheme, which reads its fields with the helpers below.

use super::{elem, Ctx, Shape};
use crate::format::Result;

/// The part's shape for structs of `n` elements in `size` bytes; `C` must be whole structs.
pub fn shape(c: &Ctx, n: usize, size: usize) -> Result<Shape> {
    if !c.cols.is_multiple_of(n) {
        return Err(c.fail(format!("{} columns are not whole blocks of {n}", c.cols)));
    }
    Ok(Shape::Exact(
        [c.lead(), &[(c.cols / n) as u64, size as u64]].concat(),
    ))
}

/// The `[R, C]` values (`R·C` of them) from structs of `size` bytes in row-major order:
/// `f(block, out)` writes the `n` elements of each.
pub fn walk(
    rows: usize,
    cols: usize,
    bytes: &[u8],
    n: usize,
    size: usize,
    mut f: impl FnMut(&[u8], &mut [f32]) -> Result<()>,
) -> Result<Vec<f32>> {
    let mut out = vec![0f32; rows * cols];
    for (block, values) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(n)) {
        f(block, values)?;
    }
    Ok(out)
}

pub fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

pub fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The `F16` at byte `at`, exactly.
pub fn f16_at(b: &[u8], at: usize) -> f32 {
    elem::f16(u16_at(b, at))
}
