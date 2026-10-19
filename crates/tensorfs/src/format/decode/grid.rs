//! Scale grids (spec §6, building blocks): which stored scale (or zero) element `(i, j)` of the `[R, C]` view
//! uses, and the stored shape that grid needs.

use super::{Ctx, Shape};
use crate::format::Result;

/// How row blocks' scales are stored; the param values of `scale_layout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLayout {
    /// `[lead…, W]`: element `i·W + b`.
    RowMajor,
    /// `[W, R]`: element `b·R + i` (GPTQ's and AWQ's `[K/g, N]`).
    Transposed,
    /// cuBLAS's tiles of 128 rows × 4 blocks, zero-padded, each matrix `[N, C]` of a stack on its own:
    /// `[…stack, 128·⌈N/128⌉, 4·⌈W/4⌉]` (ComfyUI). `at` takes one matrix's rows.
    Cublas128x4,
}

impl ScaleLayout {
    pub fn name(self) -> &'static str {
        match self {
            ScaleLayout::RowMajor => "row_major",
            ScaleLayout::Transposed => "transposed",
            ScaleLayout::Cublas128x4 => "cublas_128x4",
        }
    }
    pub fn parse(name: &str) -> Option<ScaleLayout> {
        [Self::RowMajor, Self::Transposed, Self::Cublas128x4]
            .into_iter()
            .find(|l| l.name() == name)
    }
}

/// Where element `(i, j)` finds its scale `s(i, j)`.
#[derive(Debug, Clone, Copy)]
pub enum Grid<'a> {
    /// One for the tensor.
    Tensor,
    /// `s[i]`.
    Row,
    /// `s[j]`.
    Col,
    /// Blocks of `size` consecutive elements of a row, `W = ⌈C/size⌉` to a row: `s(i, b)` with
    /// `b = ⌊j/size⌋`, or `b = index[j]` (GPTQ's `g_idx`).
    Blocks {
        size: usize,
        layout: ScaleLayout,
        index: Option<&'a [u32]>,
    },
    /// Tiles of `rows × cols` over each matrix of `height` rows (the logical shape's second-last
    /// extent), stored `[lead…, ⌈height/rows⌉, ⌈C/cols⌉]`; edge tiles may be short.
    Tiles {
        rows: usize,
        cols: usize,
        height: usize,
    },
}

impl Grid<'_> {
    /// The stored shape of a part on this grid for the context's view.
    pub fn shape(&self, c: &Ctx) -> Result<Shape> {
        Ok(match *self {
            Grid::Tensor => Shape::Count(1),
            Grid::Row => Shape::Count(c.rows as u64),
            Grid::Col => Shape::Count(c.cols as u64),
            Grid::Blocks { size, layout, .. } => {
                let w = c.cols.div_ceil(size) as u64;
                Shape::Exact(match layout {
                    ScaleLayout::RowMajor => [c.lead(), &[w]].concat(),
                    ScaleLayout::Transposed => vec![w, c.rows as u64],
                    ScaleLayout::Cublas128x4 => {
                        let (stack, n) = c.shape.split_at(c.shape.len() - 2);
                        [stack, &[n[0].div_ceil(128) * 128, w.div_ceil(4) * 4]].concat()
                    }
                })
            }
            Grid::Tiles { rows, cols, height } => {
                let (lead, c) = (&c.shape[..c.shape.len() - 2], c.shape[c.shape.len() - 1]);
                let tiles = [height.div_ceil(rows) as u64, c.div_ceil(cols as u64)];
                Shape::Exact([lead, &tiles].concat())
            }
        })
    }

    /// The index of `s(i, j)` in the stored part, for a view of `cols` columns and `rows` rows.
    pub fn at(&self, rows: usize, cols: usize, i: usize, j: usize) -> usize {
        match *self {
            Grid::Tensor => 0,
            Grid::Row => i,
            Grid::Col => j,
            Grid::Blocks {
                size,
                layout,
                index,
            } => {
                let w = cols.div_ceil(size);
                let b = index.map_or(j / size, |g| g[j] as usize);
                match layout {
                    ScaleLayout::RowMajor => i * w + b,
                    ScaleLayout::Transposed => b * rows + i,
                    ScaleLayout::Cublas128x4 => {
                        let tile = (i / 128) * w.div_ceil(4) + b / 4;
                        tile * 512 + (i % 32) * 16 + (i % 128 / 32) * 4 + b % 4
                    }
                }
            }
            Grid::Tiles {
                rows: br,
                cols: bc,
                height,
            } => {
                let (matrix, r) = (i / height, i % height);
                (matrix * height.div_ceil(br) + r / br) * cols.div_ceil(bc) + j / bc
            }
        }
    }

    /// A `g_idx` fits: one block index per column, each below `W`.
    pub fn check(&self, c: &Ctx) -> Result<()> {
        if let Grid::Blocks {
            size,
            index: Some(g),
            ..
        } = self
        {
            let w = c.cols.div_ceil(*size);
            if g.len() != c.cols || g.iter().any(|b| *b as usize >= w) {
                return Err(c.fail(format!("a block index is not one of {w} per column")));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cublas_tiles_place_every_scale_once() {
        // 130 rows × 9 blocks: two row tiles, three block tiles, every slot distinct.
        let (rows, w) = (130, 9);
        let g = Grid::Blocks {
            size: 1,
            layout: ScaleLayout::Cublas128x4,
            index: None,
        };
        let mut seen = vec![false; 256 * 12];
        for i in 0..rows {
            for b in 0..w {
                let at = g.at(rows, w, i, b);
                assert!(!seen[at], "{i} {b}");
                seen[at] = true;
            }
        }
        // Row 33, block 5: tile 1, then (33 mod 32)·16 + ⌊33/32⌋·4 + 5 mod 4.
        assert_eq!(g.at(rows, w, 33, 5), 512 + 16 + 4 + 1);
        let t = Grid::Blocks {
            size: 4,
            layout: ScaleLayout::Transposed,
            index: Some(&[1, 0, 1, 0, 1, 0, 1, 0]),
        };
        assert_eq!(t.at(3, 8, 2, 0), 3 + 2);
        assert_eq!(t.at(3, 8, 2, 1), 2);
        // Two matrices of 3 rows × 5 in 2×2 tiles: [2, 2, 3] scales; row 4 is matrix 1's row 1.
        let tiles = Grid::Tiles {
            rows: 2,
            cols: 2,
            height: 3,
        };
        assert_eq!(tiles.at(6, 5, 4, 4), 2 * 3 + 2);
        assert_eq!(tiles.at(6, 5, 2, 1), 3);
    }
}
