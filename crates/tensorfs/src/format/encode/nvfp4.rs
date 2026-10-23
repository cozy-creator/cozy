//! `nvfp4`'s encoders. `quantize` is ModelOpt's round to nearest (`NVFP4QTensor.quantize`, byte for
//! byte); `pack` lays out a calibrated producer's codes and scales. Both write either layout, a
//! global scale per tensor or per matrix, and AWQ's `pre_quant_scale`.

use super::affine::pack as pack_words;
use super::elem::{e2m1, e4m3};
use super::{encoding, Encoder, Input, OwnedPart};
use crate::format::decode::affine::{Codes, Fill, Order, Packing};
use crate::format::decode::elem::floats;
use crate::format::decode::grid::Grid;
use crate::format::{Dtype, Encoding, Error, Result, ScaleLayout};

/// Which nibble of a byte holds an F4 pair's even element: ModelOpt's low, ComfyUI's high.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder {
    LowFirst,
    HighFirst,
}

/// How nvfp4's global scales, `scale_2` and `input_scale`, apply: multiplied (ModelOpt,
/// ComfyUI) or divided, being reciprocals (compressed-tensors' `weight_global_scale` and
/// `input_global_scale`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalScale {
    Multiply,
    Divide,
}

impl NibbleOrder {
    pub fn name(self) -> &'static str {
        match self {
            NibbleOrder::LowFirst => "low_first",
            NibbleOrder::HighFirst => "high_first",
        }
    }
}

impl GlobalScale {
    pub fn name(self) -> &'static str {
        match self {
            GlobalScale::Multiply => "multiply",
            GlobalScale::Divide => "divide",
        }
    }
}

impl Encoding {
    pub fn nvfp4(global: GlobalScale, nibbles: NibbleOrder, layout: ScaleLayout) -> Encoding {
        let p = [
            ("block_size", 16u64.into()),
            ("global_scale", global.name().into()),
            ("nibble_order", nibbles.name().into()),
            ("scale_layout", layout.name().into()),
        ];
        encoding("nvfp4", &p)
    }
}

pub const NVFP4: Encoder = Encoder {
    scheme: "nvfp4",
    quantize: Some(quantize),
    pack: Some(pack),
};

const E2M1_MAX: f32 = 6.0;
const E4M3_MAX: f32 = 448.0;

fn fail(detail: impl std::fmt::Display) -> Error {
    Error::Encode(format!("nvfp4: {detail}"))
}

/// The `[R, C]` view of `[…stack, N, C]`: `R`, `C`, `N` and the stack.
struct View<'a> {
    rows: usize,
    cols: usize,
    n: usize,
    stack: &'a [u64],
}

fn view<'a>(input: &Input<'a>) -> Result<View<'a>> {
    let [stack @ .., n, c] = input.shape else {
        return Err(fail(format!(
            "shape {:?} is not rank 2 or more",
            input.shape
        )));
    };
    if !c.is_multiple_of(16) {
        return Err(fail(format!("{c} columns are not whole blocks of 16")));
    }
    let rows = stack.iter().product::<u64>() * n;
    Ok(View {
        rows: rows as usize,
        cols: *c as usize,
        n: *n as usize,
        stack,
    })
}

/// A named array as f32, `count` elements (any shape).
fn array(input: &Input, role: &str, dtypes: &[Dtype], count: &[usize]) -> Result<Option<Vec<f32>>> {
    let Some(a) = input.arrays.get(role) else {
        return Ok(None);
    };
    let n = a.shape.iter().product::<u64>() as usize;
    if !dtypes.contains(a.dtype)
        || !count.contains(&n)
        || a.dtype.byte_len(a.shape) != Some(a.bytes.len() as u64)
    {
        return Err(fail(format!("{role} is {} {:?}", a.dtype, a.shape)));
    }
    Ok(Some(floats(a.dtype, a.bytes)))
}

/// AWQ's `pre_quant_scale`, when given: stored in its own dtype.
fn pre_quant_scale(input: &Input, v: &View) -> Result<Option<Vec<f32>>> {
    let dtypes = [Dtype::F32, Dtype::BF16, Dtype::F16];
    array(input, "pre_quant_scale", &dtypes, &[v.cols])
}

/// Round to nearest as ModelOpt 0.47.0 does (`get_weights_scaling_factor_2`,
/// `get_weights_scaling_factor`, `quantize`): `g = amax / (6 · 448)` unless `scale_2` is given (one,
/// or one per matrix: a calibrated or fused group's); per block `s = E4M3(clamp(amax_b / (6 · g),
/// 2⁻⁹, 448))`, `1` for an all-zero block; then `q = E2M1(x / (s · g))`. With a `pre_quant_scale`
/// `p`, `x` is the smoothed `values[i, j] × (1 / p[j])` (AWQ's `apply_pre_quant_scale_and_smooth`).
fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let v = view(input)?;
    if input
        .encoding
        .params
        .get("global_scale")
        .and_then(|g| g.as_str())
        != Some("multiply")
    {
        return Err(fail(
            "round to nearest is ModelOpt's, global_scale multiply",
        ));
    }
    let Some(mut x) = array(input, "values", &[Dtype::F32], &[v.rows * v.cols])? else {
        return Err(fail("no values"));
    };
    let p = pre_quant_scale(input, &v)?;
    if let Some(p) = &p {
        let inv: Vec<f32> = p.iter().map(|p| 1.0 / p).collect();
        x.iter_mut()
            .enumerate()
            .for_each(|(k, x)| *x *= inv[k % v.cols]);
    }
    let matrices = (v.rows / v.n.max(1)).max(1);
    let g = match array(input, "scale_2", &[Dtype::F32], &[1, matrices])? {
        Some(g) => g,
        None => vec![x.iter().fold(0f32, |a, x| a.max(x.abs())) / (E2M1_MAX * E4M3_MAX)],
    };
    if g.iter().any(|g| !g.is_finite() || *g <= 0.0) {
        return Err(fail(format!("scale_2 {g:?} is not positive and finite")));
    }
    let (mut codes, mut scales) = (
        Vec::with_capacity(x.len()),
        Vec::with_capacity(x.len() / 16),
    );
    for (b, block) in x.as_chunks::<16>().0.iter().enumerate() {
        let g = g[if g.len() == 1 {
            0
        } else {
            b * 16 / (v.n * v.cols)
        }];
        let amax = block.iter().fold(0f32, |a, x| a.max(x.abs()));
        let s = match amax / (E2M1_MAX * g) {
            0.0 => 1.0,
            s => s.clamp(2f32.powi(-9), E4M3_MAX),
        };
        let s = e4m3(s);
        let sg = crate::format::decode::elem::e4m3(s) * g;
        // ModelOpt's sign is `x < 0`: a zero, even −0, is code 0.
        codes.extend(
            block
                .iter()
                .map(|x| if *x == 0.0 { 0 } else { e2m1(x / sg) as i32 }),
        );
        scales.push(s);
    }
    let scale_2: Vec<u8> = g.iter().flat_map(|g| g.to_le_bytes()).collect();
    layout(input, &v, &codes, &scales, scale_2, g.len())
}

/// A calibrated producer's `codes` (`U8`, an E2M1 code per element, the logical shape), its
/// row-major block `scale` (`F8_E4M3`, `[…lead, C/16]`) and `scale_2` (`F32`, one or one per
/// matrix), with its `pre_quant_scale` and `input_scale` where it has them.
pub(super) fn pack(input: &Input) -> Result<Vec<OwnedPart>> {
    let v = view(input)?;
    let a = |role: &str| {
        input
            .arrays
            .get(role)
            .ok_or_else(|| fail(format!("no {role}")))
    };
    let (codes, scale, g) = (a("codes")?, a("scale")?, a("scale_2")?);
    let numel = |shape: &[u64]| shape.iter().product::<u64>() as usize;
    if *codes.dtype != Dtype::U8
        || codes.shape != input.shape
        || codes.bytes.iter().any(|q| *q > 15)
    {
        return Err(fail(format!(
            "codes are {} {:?}, not E2M1 codes of the logical shape",
            codes.dtype, codes.shape
        )));
    }
    if *scale.dtype != Dtype::F8_E4M3 || numel(scale.shape) != v.rows * v.cols / 16 {
        return Err(fail(format!("scale is {} {:?}", scale.dtype, scale.shape)));
    }
    let matrices = (v.rows / v.n.max(1)).max(1);
    if *g.dtype != Dtype::F32 || ![1, matrices].contains(&numel(g.shape)) {
        return Err(fail(format!("scale_2 is {} {:?}", g.dtype, g.shape)));
    }
    pre_quant_scale(input, &v)?;
    let codes: Vec<i32> = codes.bytes.iter().map(|q| *q as i32).collect();
    layout(
        input,
        &v,
        &codes,
        scale.bytes,
        g.bytes.to_vec(),
        numel(g.shape),
    )
}

/// The parts: codes packed by `nibble_order`, scales stored by `scale_layout` (cuBLAS tiles per
/// matrix), `scale_2` as `[]` or `[…stack]`, then `input_scale` and `pre_quant_scale` as given.
fn layout(
    input: &Input,
    v: &View,
    codes: &[i32],
    scales: &[u8],
    scale_2: Vec<u8>,
    globals: usize,
) -> Result<Vec<OwnedPart>> {
    let text = |key: &str| input.encoding.params.get(key).and_then(|t| t.as_str());
    let order = match text("nibble_order") {
        Some("low_first") => Order::LowFirst,
        Some("high_first") => Order::HighFirst,
        other => return Err(fail(format!("nibble_order {other:?}"))),
    };
    let tiled = match text("scale_layout") {
        Some("row_major") => false,
        Some("cublas_128x4") => true,
        other => return Err(fail(format!("scale_layout {other:?}"))),
    };
    let packing = Packing {
        bits: 4,
        word: 8,
        transposed: false,
        axis: 1,
        fill: Fill::Word,
        order,
        codes: Codes::Unsigned(0),
    };
    array(
        input,
        "input_scale",
        &[Dtype::F32],
        &[1, v.rows / v.n.max(1)],
    )?;
    let w = v.cols / 16;
    let (scale_shape, scale_bytes) = match tiled {
        false => (
            [&input.shape[..input.shape.len() - 1], &[w as u64]].concat(),
            scales.to_vec(),
        ),
        true => {
            let (rows, blocks) = (v.n.div_ceil(128) * 128, w.div_ceil(4) * 4);
            let grid = Grid::Blocks {
                size: 1,
                layout: ScaleLayout::Cublas128x4,
                index: None,
            };
            let mut out = vec![0u8; (v.rows / v.n.max(1)) * rows * blocks];
            for (i, row) in scales.chunks_exact(w.max(1)).enumerate() {
                for (b, s) in row.iter().enumerate() {
                    out[grid.at(v.rows, w, v.n, i, b)] = *s;
                }
            }
            ([v.stack, &[rows as u64, blocks as u64]].concat(), out)
        }
    };
    let part = |role: &str, dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>| OwnedPart {
        role: role.into(),
        dtype,
        shape,
        bytes,
    };
    let mut parts = vec![
        part(
            "weight",
            Dtype::F4,
            input.shape.to_vec(),
            pack_words(&packing, codes, v.rows, v.cols)?,
        ),
        part("scale", Dtype::F8_E4M3, scale_shape, scale_bytes),
        part(
            "scale_2",
            Dtype::F32,
            if globals == 1 {
                vec![]
            } else {
                v.stack.to_vec()
            },
            scale_2,
        ),
    ];
    for role in ["input_scale", "pre_quant_scale"] {
        if let Some(a) = input.arrays.get(role) {
            parts.push(part(
                role,
                a.dtype.clone(),
                a.shape.to_vec(),
                a.bytes.to_vec(),
            ));
        }
    }
    Ok(parts)
}
