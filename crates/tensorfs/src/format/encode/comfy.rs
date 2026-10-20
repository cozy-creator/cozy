//! `comfy.*` encoders: comfy-kitchen 0.2.39's eager quantizers (`quantize_w4a8_int8_weight`,
//! `quantize_convrot_w4a4_weight`), bit for bit with PyTorch 2.14's CPU kernels on one thread.
//! Kitchen rotates in the weight's dtype `L`, so the rotated weight is rounded to `L` before it is
//! quantized; the rotation is computed in f32 and agrees with kitchen's matmul whenever that sum is
//! exact (always for bf16 weights of ordinary range). The W4A8 codebook is kitchen's frozen
//! Lloyd-Max table unless a kurtosis gate finds the rotated groups heavy-tailed, then a Lloyd-Max
//! fit; its `pow(4)` is `(z²)²` here (PyTorch uses Sleef), so a layer within ~1e−6 of the gate
//! may decide differently. W4A8 refines group scales by two ALS passes; W6A8 then searches each
//! group's scale among its stored-dtype neighbours.

use super::{elem, encoding, Encoder, Input, OwnedPart};
use crate::format::decode::{bf16, check, f16, SCHEMES};
use crate::format::{Dtype, Encoding, Error, Result, Value};

pub const ASYM_W4A8_INT8: Encoder = Encoder {
    scheme: "comfy.asym_w4a8_int8",
    quantize: Some(|i| quantize_with(i, &Kitchen::default())),
    pack: None,
};

pub const W6A8_INT8: Encoder = Encoder {
    scheme: "comfy.w6a8_int8",
    quantize: Some(|i| quantize_with(i, &Kitchen::default())),
    pack: None,
};

pub const CONVROT_W4A4: Encoder = Encoder {
    scheme: "comfy.convrot_w4a4",
    quantize: Some(|i| quantize_with(i, &Kitchen::default())),
    pack: None,
};

impl Encoding {
    pub fn comfy_asym_w4a8_int8(group_size: u64, rotation_size: u64) -> Encoding {
        let p = [
            ("group_size", group_size.into()),
            ("rotation_size", rotation_size.into()),
        ];
        encoding("comfy.asym_w4a8_int8", &p)
    }
    pub fn comfy_w6a8_int8(group_size: u64, rotation_size: u64) -> Encoding {
        let p = [
            ("group_size", group_size.into()),
            ("rotation_size", rotation_size.into()),
        ];
        encoding("comfy.w6a8_int8", &p)
    }
    pub fn comfy_convrot_w4a4(rotation_size: u64) -> Encoding {
        encoding(
            "comfy.convrot_w4a4",
            &[("rotation_size", rotation_size.into())],
        )
    }
}

/// Kitchen's quantizer options beside the encoding's params; `default()` is what ComfyUI
/// writes (a codebook, symmetric, `F8_E4M3` group scales, W6A8's scale search).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kitchen {
    /// 4-bit: a Lloyd-Max codebook (`weight_codebook`); else uniform `q − 8`.
    pub codebook: bool,
    /// 4-bit: `false` is kitchen's asymmetric min/max grid with a `weight_correction`.
    /// W6A8 refuses `false`, because that format has no correction part.
    pub symmetric: bool,
    /// `weight_s_rel` in `F32` rather than `F8_E4M3`.
    pub f32_scales: bool,
    /// 6-bit: the grid-aware group-scale search (kitchen's requantize skips it).
    pub scale_search: bool,
}

impl Default for Kitchen {
    fn default() -> Kitchen {
        Kitchen {
            codebook: true,
            symmetric: true,
            f32_scales: false,
            scale_search: true,
        }
    }
}

/// The frozen Lloyd-Max levels of a group-normalized Gaussian (kitchen's `_FIXED_LUT`).
const LUT: [f32; 16] = [
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
];

/// The parts of `input.encoding` from its values with kitchen's `options` (ignored by
/// `comfy.convrot_w4a4`, which has none).
pub fn quantize_with(input: &Input, options: &Kitchen) -> Result<Vec<OwnedPart>> {
    let scheme = input.encoding.scheme.as_str();
    let fail = |d: String| Error::Encode(format!("{scheme}: {d}"));
    if !matches!(
        scheme,
        "comfy.asym_w4a8_int8" | "comfy.w6a8_int8" | "comfy.convrot_w4a4"
    ) {
        return Err(Error::UnknownScheme(scheme.into()));
    }
    let schema = SCHEMES
        .iter()
        .find(|s| s.name == scheme)
        .ok_or_else(|| Error::UnknownScheme(scheme.into()))?;
    if let Some(key) = input
        .encoding
        .params
        .keys()
        .find(|k| !schema.params.iter().any(|p| p.key == *k))
    {
        return Err(fail(format!("unknown param {key:?}")));
    }
    let uint = |key: &str| match input.encoding.params.get(key).and_then(Value::as_u64) {
        Some(v) => usize::try_from(v).map_err(|_| fail(format!("param {key} is too large"))),
        None => Err(fail(format!("param {key} is missing or not an integer"))),
    };
    let (&[ref lead @ .., c], true) = (input.shape, input.shape.len() >= 2) else {
        return Err(fail(format!(
            "logical shape {:?} is not rank 2 or more",
            input.shape
        )));
    };
    if !matches!(input.dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16) {
        return Err(fail(format!(
            "logical dtype {} is not F32, BF16 or F16",
            input.dtype
        )));
    }
    let want = Dtype::F32
        .byte_len(input.shape)
        .ok_or_else(|| fail("values shape overflows its F32 byte count".into()))?;
    let actual = input
        .arrays
        .get("values")
        .ok_or_else(|| fail("no values".into()))?
        .bytes
        .len();
    if actual as u64 != want {
        return Err(fail(format!("values have {actual} bytes, not {want}")));
    }
    if want == 0 {
        return Err(fail("empty values cannot be quantized".into()));
    }
    let values = input.values()?;
    let b = uint("rotation_size")?;
    if !(4..=65536).contains(&b) || !b.is_power_of_two() || b.trailing_zeros() % 2 == 1 {
        return Err(fail(format!("rotation_size {b} is not a power of 4")));
    }
    let k = usize::try_from(c).map_err(|_| fail("column count is too large".into()))?;
    let w = Weight {
        rows: values.len() / k,
        k,
        lead: lead.to_vec(),
        dtype: input.dtype.clone(),
    };
    if !k.is_multiple_of(b) {
        return Err(fail(format!("{k} columns are not whole blocks of {b}")));
    }
    let group = match scheme {
        "comfy.convrot_w4a4" if k.is_multiple_of(64) => None,
        "comfy.convrot_w4a4" => {
            return Err(fail(format!("{k} columns are not whole groups of 64")))
        }
        _ => {
            let bits = if scheme == "comfy.w6a8_int8" { 6 } else { 4 };
            let g = uint("group_size")?;
            let ok = match bits {
                6 => g >= 16 && g.is_multiple_of(16) && k.is_multiple_of(32),
                _ => g >= 4 && (16 % g == 0 || g.is_multiple_of(16)) && k.is_multiple_of(16),
            };
            if !ok || !k.is_multiple_of(g) {
                return Err(fail(format!("group_size {g} does not fit {k} columns")));
            }
            if values.len() as u64 >= 1 << 32 {
                return Err(fail("2^32 or more elements".into()));
            }
            if bits == 6 && !options.symmetric {
                return Err(fail("W6A8 has no asymmetric correction option".into()));
            }
            Some((g, bits))
        }
    };
    let rotated = rotate(&values, b, &w.dtype);
    if rotated.iter().any(|v| !v.is_finite()) {
        return Err(fail("rotation is not finite in the logical dtype".into()));
    }
    let parts = match group {
        None => w4a4(&w, &rotated),
        Some((g, bits)) => grouped(&w, &rotated, g, bits, options),
    };
    for p in &parts {
        if p.dtype.byte_len(&p.shape) != Some(p.bytes.len() as u64) {
            return Err(fail(format!("part {} has an invalid byte count", p.role)));
        }
        if p.dtype != Dtype::I8
            && crate::format::decode::elem::floats(&p.dtype, &p.bytes)
                .iter()
                .any(|v| !v.is_finite())
        {
            return Err(fail(format!("part {} is not finite", p.role)));
        }
    }
    let views = parts
        .iter()
        .map(|p| {
            (
                p.role.as_str(),
                super::PartBytes {
                    dtype: &p.dtype,
                    shape: &p.shape,
                    bytes: &p.bytes,
                },
            )
        })
        .collect();
    check(input.dtype, input.shape, input.encoding, &views)
        .map_err(|e| fail(format!("invalid encoded parts: {e}")))?;
    Ok(parts)
}

struct Weight {
    rows: usize,
    k: usize,
    lead: Vec<u64>,
    dtype: Dtype,
}

/// `round_L`, back in f32.
fn round_l(dtype: &Dtype, x: f32) -> f32 {
    match dtype {
        Dtype::BF16 => f32::from_bits((bf16(x) as u32) << 16),
        Dtype::F16 => crate::format::decode::elem::f16(f16(x)),
        _ => x,
    }
}

/// Each row's blocks of `b` times `H_B / 2^k` (`H₄` along each base-4 digit, in f32), rounded to
/// `L`: kitchen's `W @ H.T` in the weight's dtype.
fn rotate(x: &[f32], b: usize, dtype: &Dtype) -> Vec<f32> {
    let unit = crate::format::decode::elem::pow2(-(b.trailing_zeros() as i32 / 2));
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks(b) {
        let mut y = block.to_vec();
        let mut stride = 1;
        while stride < b {
            for base in (0..b).filter(|i| (i / stride) % 4 == 0) {
                let [p, q, r, s] = [0, 1, 2, 3].map(|j| y[base + j * stride]);
                let v = [p + q + r - s, p + q - r + s, p - q + r + s, q - p + r + s];
                for (j, v) in v.into_iter().enumerate() {
                    y[base + j * stride] = v;
                }
            }
            stride *= 4;
        }
        out.extend(y.iter().map(|v| round_l(dtype, v * unit)));
    }
    out
}

fn part(role: &str, dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>) -> OwnedPart {
    OwnedPart {
        role: role.into(),
        dtype,
        shape,
        bytes,
    }
}

fn f32_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn l_bytes(dtype: &Dtype, x: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::BF16 => x.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect(),
        Dtype::F16 => x.iter().flat_map(|v| f16(*v).to_le_bytes()).collect(),
        _ => f32_bytes(x),
    }
}

/// `quantize_convrot_w4a4_weight`: a symmetric int4 scale per row, `round_L(amax / 7)`, codes
/// `round(round_L(x / s))` in `±7` (kitchen computes in `L`).
fn w4a4(w: &Weight, x: &[f32]) -> Vec<OwnedPart> {
    let tiny = round_l(&w.dtype, 1e-10);
    let mut codes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(w.rows);
    for row in x.chunks(w.k.max(1)) {
        let amax = row.iter().fold(0f32, |a, v| a.max(v.abs())).max(tiny);
        let s = round_l(&w.dtype, amax / 7.0);
        scales.push(s);
        codes.extend(
            row.iter()
                .map(|v| round_l(&w.dtype, v / s).round_ties_even().clamp(-7.0, 7.0) as i32),
        );
    }
    vec![
        part(
            "weight",
            Dtype::I8,
            packed_shape(w, 4),
            pack(&codes, w.k, 4),
        ),
        part(
            "weight_scale",
            Dtype::F32,
            w.lead.clone(),
            f32_bytes(&scales),
        ),
    ]
}

fn packed_shape(w: &Weight, bits: usize) -> Vec<u64> {
    [&w.lead[..], &[(w.k * bits / 8) as u64]].concat()
}

/// Kitchen's storage: per row, `K/2` bytes of nibbles (even column low), then at 6 bits `K/4`
/// bytes of each code's high 2 bits (column `t` at bits `2(t mod 4)` of byte `t/4`).
fn pack(q: &[i32], k: usize, bits: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(q.len() * bits / 8);
    for row in q.chunks(k.max(1)) {
        out.extend(
            row.chunks(2)
                .map(|p| ((p[0] & 15) | (p[1] & 15) << 4) as u8),
        );
        if bits == 6 {
            out.extend(
                row.chunks(4)
                    .map(|p| (0..4).fold(0u8, |a, j| a | (((p[j] >> 4) & 3) << (2 * j)) as u8)),
            );
        }
    }
    out
}

/// `_quantize_w4a8_chunked` for 4- and 6-bit codes (one codebook for the whole tensor).
fn grouped(w: &Weight, x: &[f32], g: usize, bits: usize, o: &Kitchen) -> Vec<OwnedPart> {
    let (n, k, groups) = (w.rows, w.k, w.k / g);
    let codebook = (bits == 4 && o.symmetric && o.codebook).then(|| decide_codebook(x, n, k, g));
    let w6: Vec<f32> = (-31..=31).map(|v| v as f32).collect();
    let table: Option<&[f32]> = match (&codebook, bits) {
        (_, 6) => Some(&w6),
        (Some(cb), _) => Some(cb),
        _ => None,
    };
    let (mut scale, mut q) = (vec![0f32; n * groups], vec![0i32; n * k]);
    let mut corr = vec![0f32; if o.symmetric { 0 } else { n * groups }];
    let mut shifted = vec![0f32; n * k];
    for (at, grp) in x.chunks(g).enumerate() {
        let (sc, qs, sh) = (
            &mut scale[at],
            &mut q[at * g..][..g],
            &mut shifted[at * g..][..g],
        );
        let amax = grp.iter().fold(0f32, |a, v| a.max(v.abs()));
        match table {
            Some(t) => {
                let assign = |s: f32, qs: &mut [i32]| {
                    for (q, v) in qs.iter_mut().zip(grp) {
                        *q = match bits {
                            6 => (v / s).round_ties_even().clamp(-31.0, 31.0) as i32 + 31,
                            _ => nearest(t, v / s),
                        };
                    }
                };
                *sc = if bits == 6 {
                    (amax / 31.0).max(1e-8)
                } else {
                    amax.max(1e-8)
                };
                assign(*sc, qs);
                for _ in 0..2 {
                    let l: Vec<f32> = qs.iter().map(|q| t[*q as usize]).collect();
                    let num: Vec<f32> = grp.iter().zip(&l).map(|(a, b)| a * b).collect();
                    let den: Vec<f32> = l.iter().map(|v| v * v).collect();
                    *sc = (tsum(&num) / tsum(&den).max(1e-8)).max(1e-8);
                    assign(*sc, qs);
                }
                for (s, q) in sh.iter_mut().zip(qs.iter()) {
                    *s = t[*q as usize] * *sc;
                }
            }
            None if o.symmetric => {
                *sc = (amax / 7.0).max(1e-8);
                for ((q, s), v) in qs.iter_mut().zip(sh.iter_mut()).zip(grp) {
                    let signed = (v / *sc).round_ties_even().clamp(-8.0, 7.0);
                    (*q, *s) = (signed as i32 + 8, signed * *sc);
                }
            }
            None => {
                let mn = grp.iter().fold(f32::INFINITY, |a, v| a.min(*v));
                let mx = grp.iter().fold(f32::NEG_INFINITY, |a, v| a.max(*v));
                *sc = ((mx - mn) / 15.0).max(1e-8);
                for ((q, s), v) in qs.iter_mut().zip(sh.iter_mut()).zip(grp) {
                    *q = ((v - mn) / *sc).round_ties_even().clamp(0.0, 15.0) as i32;
                    *s = (*q - 8) as f32 * *sc;
                }
                let (i, j) = (at / groups, at % groups);
                corr[j * n + i] = round_l(&w.dtype, 8.0 * *sc + mn);
            }
        }
    }
    let s_ch: Vec<f32> = (shifted.chunks(k.max(1)))
        .map(|r| (r.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0).max(1e-8))
        .collect();
    let mut s_rel: Vec<f32> = (0..n * groups)
        .map(|at| scale[at] / s_ch[at / groups])
        .collect();
    if !o.f32_scales {
        s_rel = s_rel.iter().map(|v| e4m3_value(elem::e4m3(*v))).collect();
    }
    if let Some(t) = table {
        for at in 0..n * groups {
            let target: Vec<f32> = x[at * g..][..g]
                .iter()
                .map(|v| v / s_ch[at / groups])
                .collect();
            let qs = &mut q[at * g..][..g];
            if bits == 6 && o.scale_search {
                s_rel[at] = search(t, s_rel[at], &target, qs, o.f32_scales);
            } else {
                assign_grid(&levels(t, s_rel[at]), &target, qs);
            }
            if bits == 6 {
                qs.iter_mut().for_each(|q| *q += 1);
            }
        }
    }
    let rel_shape = [&w.lead[..], &[groups as u64]].concat();
    let mut parts = vec![
        part(
            "weight",
            Dtype::I8,
            packed_shape(w, bits),
            pack(&q, k, bits),
        ),
        match o.f32_scales {
            true => part("weight_s_rel", Dtype::F32, rel_shape, f32_bytes(&s_rel)),
            false => {
                let b = s_rel.iter().map(|v| elem::e4m3(*v)).collect();
                part("weight_s_rel", Dtype::F8_E4M3, rel_shape, b)
            }
        },
        part(
            "weight_s_channel",
            Dtype::F32,
            w.lead.clone(),
            f32_bytes(&s_ch),
        ),
    ];
    if let Some(cb) = &codebook {
        parts.push(part("weight_codebook", Dtype::F32, vec![16], f32_bytes(cb)));
    }
    if !o.symmetric {
        let shape = vec![groups as u64, n as u64];
        parts.push(part(
            "weight_correction",
            w.dtype.clone(),
            shape,
            l_bytes(&w.dtype, &corr),
        ));
    }
    parts
}

fn e4m3_value(b: u8) -> f32 {
    crate::format::decode::elem::e4m3(b)
}

/// `_grid_levels`: the int8 value each level decodes to, `clamp(round(level × s_rel), ±127)`.
fn levels(table: &[f32], s: f32) -> Vec<f32> {
    let lv = table
        .iter()
        .map(|t| (t * s).round_ties_even().clamp(-127.0, 127.0));
    lv.collect()
}

/// ATen's `searchsorted` (left): a binary search, whether or not `sorted` is.
fn lower_bound(sorted: &[f32], v: f32) -> usize {
    let (mut lo, mut hi) = (0, sorted.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if sorted[mid] >= v {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The nearer of the insertion point's two neighbours, the lower on a tie (`_assign_codes`).
fn nearest(table: &[f32], v: f32) -> i32 {
    let last = table.len() - 1;
    let pos = lower_bound(table, v);
    let (lo, hi) = (pos.saturating_sub(1).min(last), pos.min(last));
    match (v - table[hi]).abs() < (v - table[lo]).abs() {
        true => hi as i32,
        false => lo as i32,
    }
}

fn assign_grid(levels: &[f32], target: &[f32], q: &mut [i32]) {
    for (q, t) in q.iter_mut().zip(target) {
        *q = nearest(levels, *t);
    }
}

/// `_search_group_scale`: the scale and its two stored-dtype neighbours (`±1` on the `F8_E4M3`
/// code, or `×(1 ∓ 0.03)` in `F32`), keeping the first with the least squared error.
fn search(table: &[f32], s: f32, target: &[f32], q: &mut [i32], f32_scales: bool) -> f32 {
    let candidates = match f32_scales {
        true => [s, s * (1.0f64 - 0.03) as f32, s * (1.0f64 + 0.03) as f32],
        false => {
            let raw = elem::e4m3(s);
            let down = raw.wrapping_sub(1).max(1);
            let up = raw.wrapping_add(1).min(0x7e);
            [s, e4m3_value(down), e4m3_value(up)]
        }
    };
    let mut best: Option<(f32, f32, Vec<i32>)> = None;
    for c in candidates {
        let lv = levels(table, c);
        let mut idx = vec![0; target.len()];
        assign_grid(&lv, target, &mut idx);
        let sq: Vec<f32> = (idx.iter().zip(target))
            .map(|(i, t)| (lv[*i as usize] - t) * (lv[*i as usize] - t))
            .collect();
        let err = tsum(&sq);
        if best.as_ref().is_none_or(|(e, _, _)| err < *e) {
            best = Some((err, c, idx));
        }
    }
    let (_, c, idx) = best.expect("three candidates");
    q.copy_from_slice(&idx);
    c
}

/// `_decide_codebook`: the codebook of a row sample spanning the tensor, from each group's
/// values over its absolute maximum.
fn decide_codebook(x: &[f32], n: usize, k: usize, g: usize) -> Vec<f32> {
    let max_rows = ((1usize << 22) / k.max(1)).max(1);
    let rows: Vec<usize> = match n > max_rows {
        true => linspace((n - 1) as f32, max_rows)
            .iter()
            .map(|v| *v as usize)
            .collect(),
        false => (0..n).collect(),
    };
    let mut normalized = Vec::with_capacity(rows.len() * k);
    for r in rows {
        for grp in x[r * k..][..k].chunks(g) {
            let amax = grp.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-8);
            normalized.extend(grp.iter().map(|v| v / amax));
        }
    }
    codebook_for(&normalized)
}

/// `_codebook_for`: the frozen table when the normalized values' excess kurtosis (of up to 2^19
/// sampled) is at most −0.1, else a Lloyd-Max fit.
fn codebook_for(normalized: &[f32]) -> Vec<f32> {
    let sample: Vec<f32> = match normalized.len() > 1 << 19 {
        true => randint(normalized.len(), 1 << 19)
            .map(|i| normalized[i])
            .collect(),
        false => normalized.to_vec(),
    };
    let len = sample.len() as f32;
    let mean = tsum(&sample) / len;
    let m = mean as f64;
    let ss = sample
        .iter()
        .fold(0f64, |a, v| a + (*v as f64 - m) * (*v as f64 - m));
    let sd = (ss / (sample.len() as f64 - 1.0)).sqrt() as f32 + 1e-9;
    let p: Vec<f32> = (sample.iter())
        .map(|v| (v - mean) / sd)
        .map(|z| (z * z) * (z * z))
        .collect();
    match ((tsum(&p) / len - 3.0) as f64) <= -0.1 {
        true => LUT.to_vec(),
        false => fit(normalized),
    }
}

/// `_fit_codebook`: Lloyd-Max over up to 300 000 sampled values, from their quantiles, 25 passes.
fn fit(normalized: &[f32]) -> Vec<f32> {
    let s: Vec<f32> = match normalized.len() > 300_000 {
        true => randint(normalized.len(), 300_000)
            .map(|i| normalized[i])
            .collect(),
        false => normalized.to_vec(),
    };
    let mut cb = quantile(&s, &linspace(1.0, 16));
    for _ in 0..25 {
        let mut members = vec![Vec::new(); 16];
        for v in &s {
            let d = |j: usize| (v - cb[j]).abs();
            let best = (1..16).fold(0, |b, j| if d(j) < d(b) { j } else { b });
            members[best].push(*v);
        }
        for (c, m) in cb.iter_mut().zip(&members) {
            if !m.is_empty() {
                *c = tsum(m) / m.len() as f32;
            }
        }
    }
    cb
}

/// `torch.quantile`, linear: ranks `q·(n − 1)` in f64, then ATen's vectorized lerp,
/// `fma(w or w − 1, hi − lo, lo or hi)`.
fn quantile(x: &[f32], q: &[f32]) -> Vec<f32> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f32::total_cmp);
    q.iter()
        .map(|q| {
            let rank = *q as f64 * (sorted.len() - 1) as f64;
            let (below, above) = (rank as usize, rank.ceil() as usize);
            let w = (rank - below as f64) as f32;
            let (lo, hi) = (sorted[below], sorted[above]);
            match w.abs() < 0.5 {
                true => w.mul_add(hi - lo, lo),
                false => (w - 1.0).mul_add(hi - lo, hi),
            }
        })
        .collect()
}

/// `torch.linspace(0, end, steps)` in f32: `step·i` (or `end − step·(steps − 1 − i)` past half)
/// fused, as the compiled kernel computes it.
fn linspace(end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![0.0];
    }
    let step = end / (steps - 1) as f32;
    (0..steps)
        .map(|i| match i < steps / 2 {
            true => step.mul_add(i as f32, 0.0),
            false => (-step).mul_add((steps - i - 1) as f32, end),
        })
        .collect()
}

/// `torch.randint(0, n, (count,))` from a CPU generator seeded 0 (MT19937; `n < 2^32`).
fn randint(n: usize, count: usize) -> impl Iterator<Item = usize> {
    let mut mt = [0u32; 624];
    for j in 1..624 {
        mt[j] = 1812433253u32
            .wrapping_mul(mt[j - 1] ^ (mt[j - 1] >> 30))
            .wrapping_add(j as u32);
    }
    let mut i = 624;
    (0..count).map(move |_| {
        if i == 624 {
            for k in 0..624 {
                let y = (mt[k] & 0x8000_0000) | (mt[(k + 1) % 624] & 0x7fff_ffff);
                mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 == 1 { 0x9908_b0df } else { 0 };
            }
            i = 0;
        }
        let mut y = mt[i];
        i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y as usize % n
    })
}

/// PyTorch's CPU `sum` of a contiguous run on one thread: 8-lane vectors (the AVX2 kernel, which
/// AVX512 machines also run) through ATen's cascade, `SumKernel.cpp` `vectorized_inner_sum`.
fn tsum(x: &[f32]) -> f32 {
    if x.len() < 8 {
        return row_sum(&|i| x[i], x.len());
    }
    let nv = x.len() / 8;
    let lanes = row_sum(
        &|v| -> [f32; 8] { x[8 * v..8 * v + 8].try_into().expect("8") },
        nv,
    );
    let tail = x[nv * 8..].iter().fold(0f32, |a, v| a + v);
    lanes.iter().fold(tail, |a, v| a + v)
}

trait Lanes: Copy {
    const ZERO: Self;
    fn add(self, o: Self) -> Self;
}

impl Lanes for f32 {
    const ZERO: f32 = 0.0;
    fn add(self, o: f32) -> f32 {
        self + o
    }
}

impl Lanes for [f32; 8] {
    const ZERO: [f32; 8] = [0.0; 8];
    fn add(self, o: [f32; 8]) -> [f32; 8] {
        std::array::from_fn(|l| self[l] + o[l])
    }
}

fn row_sum<T: Lanes>(load: &impl Fn(usize) -> T, size: usize) -> T {
    let rows = size / 4;
    let mut p = multi_row_sum(load, rows);
    for i in rows * 4..size {
        p[0] = p[0].add(load(i));
    }
    for k in 1..4 {
        p[0] = p[0].add(p[k]);
    }
    p[0]
}

/// Rows of 4 summed into four levels of accumulators, a level carried up every `2^power` rows.
fn multi_row_sum<T: Lanes>(load: &impl Fn(usize) -> T, size: usize) -> [T; 4] {
    let log2 = if size <= 2 {
        1
    } else {
        (usize::BITS - (size - 1).leading_zeros()) as usize
    };
    let power = (log2 / 4).max(4);
    let (step, mask) = (1usize << power, (1usize << power) - 1);
    let add = |a: [T; 4], b: [T; 4]| std::array::from_fn(|k| a[k].add(b[k]));
    let row = |i: usize| std::array::from_fn(|k| load(4 * i + k));
    let mut acc = [[T::ZERO; 4]; 4];
    let mut i = 0;
    while i + step <= size {
        for _ in 0..step {
            acc[0] = add(acc[0], row(i));
            i += 1;
        }
        for j in 1..4 {
            acc[j] = add(acc[j], acc[j - 1]);
            acc[j - 1] = [T::ZERO; 4];
            if i & (mask << (j * power)) != 0 {
                break;
            }
        }
    }
    for i in i..size {
        acc[0] = add(acc[0], row(i));
    }
    for j in 1..4 {
        acc[0] = add(acc[0], acc[j]);
    }
    acc[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values PyTorch 2.14.1 (CPU, one thread) gives for the same calls.
    #[test]
    fn torch_pieces_match_pytorch() {
        let r: Vec<usize> = randint(1_000_003, 5).collect();
        assert_eq!(r, TORCH_RANDINT);
        let l = linspace(25599.0, 819);
        assert_eq!(
            [l[1], l[409], l[410], l[818]].map(f32::to_bits),
            TORCH_LINSPACE
        );
        let x: Vec<f32> = (0..70001)
            .map(|i| ((i * 7919 % 10007) as f32 - 5003.0) * 1e-3)
            .collect();
        assert_eq!(tsum(&x).to_bits(), TORCH_SUM);
    }

    const TORCH_RANDINT: [usize; 5] = [128973, 240601, 705720, 82882, 841199];
    const TORCH_LINSPACE: [u32; 4] = [0x41fa5b62, 0x4647fe00, 0x46487b2e, 0x46c7fe00];
    const TORCH_SUM: u32 = 0x40c6b800;
}
