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
#[derive(Debug, Clone)]
pub enum Grid {
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
        index: Option<Vec<u32>>,
    },
    /// Tiles of `rows × cols` over each matrix (`height` rows, the logical shape's second-last
    /// extent), stored `[lead…, ⌈height/rows⌉, ⌈C/cols⌉]`; edge tiles may be short.
    Tiles { rows: usize, cols: usize },
}

impl Grid {
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
            Grid::Tiles { rows, cols } => {
                let (lead, n) = c.shape.split_at(c.shape.len() - 2);
                let tiles = [n[0].div_ceil(rows as u64), n[1].div_ceil(cols as u64)];
                Shape::Exact([lead, &tiles].concat())
            }
        })
    }

    /// The index of `s(i, j)` in the stored part, for a view of `rows × cols` whose matrices have
    /// `height` rows each (`cublas_128x4` and tiles restart with each matrix).
    pub fn at(&self, rows: usize, cols: usize, height: usize, i: usize, j: usize) -> usize {
        let (m, r) = (i / height, i % height);
        match self {
            Grid::Tensor => 0,
            Grid::Row => i,
            Grid::Col => j,
            Grid::Blocks {
                size,
                layout,
                index,
            } => {
                let w = cols.div_ceil(*size);
                let b = index.as_ref().map_or(j / size, |g| g[j] as usize);
                match layout {
                    ScaleLayout::RowMajor => i * w + b,
                    ScaleLayout::Transposed => b * rows + i,
                    ScaleLayout::Cublas128x4 => {
                        let matrix = height.div_ceil(128) * 128 * w.div_ceil(4) * 4;
                        let tile = (r / 128) * w.div_ceil(4) + b / 4;
                        m * matrix + tile * 512 + (r % 32) * 16 + (r % 128 / 32) * 4 + b % 4
                    }
                }
            }
            Grid::Tiles { rows: br, cols: bc } => {
                (m * height.div_ceil(*br) + r / br) * cols.div_ceil(*bc) + j / bc
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
