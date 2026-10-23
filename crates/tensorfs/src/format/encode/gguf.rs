//! `gguf.<type>` encoders: a port of ggml's reference quantizers (`quantize_row_<type>_ref`,
//! `ggml-quants.c` at llama.cpp ff5a233, MIT, © the ggml authors), bit-exact with ggml built for
//! x86-64 without fused multiply-adds: the same f32 operations in the same order, ggml's own
//! f16 conversions, and C's float-to-int casts as x86-64 performs them. Proven against the C on
//! some 3.5 M blocks per type.

use super::{Encoder, Input, OwnedPart};
use crate::format::decode::gguf::{blocks_shape, Type, K};
use crate::format::{Dtype, Error, Result};

macro_rules! encoder {
    ($name:literal) => {
        Encoder {
            scheme: concat!("gguf.", $name),
            quantize: Some(quantize),
            pack: None,
        }
    };
}

pub const Q4_0: Encoder = encoder!("q4_0");
pub const Q4_1: Encoder = encoder!("q4_1");
pub const Q5_0: Encoder = encoder!("q5_0");
pub const Q5_1: Encoder = encoder!("q5_1");
pub const Q8_0: Encoder = encoder!("q8_0");
pub const Q2_K: Encoder = encoder!("q2_k");
pub const Q3_K: Encoder = encoder!("q3_k");
pub const Q4_K: Encoder = encoder!("q4_k");
pub const Q5_K: Encoder = encoder!("q5_k");
pub const Q6_K: Encoder = encoder!("q6_k");
pub const MXFP4: Encoder = encoder!("mxfp4");
pub const NVFP4: Encoder = encoder!("nvfp4");
pub const Q1_0: Encoder = encoder!("q1_0");
pub const Q2_0: Encoder = encoder!("q2_0");

/// `blocks` from `arrays["values"]` (`F32`, the logical shape), each struct as ggml's reference
/// quantizer writes it. Non-finite values are refused: ggml's result for them depends on the
/// compiler. `gguf.nvfp4` gets no `scale` (ggml's quantizer has none).
fn quantize(input: &Input) -> Result<Vec<OwnedPart>> {
    let scheme = input.encoding.scheme.as_str();
    let fail = |detail: String| Error::Encode(format!("{scheme}: {detail}"));
    let ty = Type::by_name(&scheme["gguf.".len()..]).expect("every row names a ggml type");
    let params = &input.encoding.params;
    if params.len() != 1 || params.get("block_size").and_then(|v| v.as_u64()) != Some(ty.block) {
        return Err(fail(format!(
            "params {params:?} are not {{block_size: {}}}",
            ty.block
        )));
    }
    let values = match input.arrays.get("values") {
        Some(v) if *v.dtype == Dtype::F32 && v.shape == input.shape => v,
        _ => return Err(fail(format!("values: F32 {:?}", input.shape))),
    };
    let Some(shape) = blocks_shape(ty, input.shape) else {
        return Err(fail(format!(
            "{:?} is not whole blocks of {}",
            input.shape, ty.block
        )));
    };
    let x: Vec<f32> = (values.bytes.as_chunks::<4>().0.iter())
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if let Some(v) = x.iter().find(|v| !v.is_finite()) {
        return Err(fail(format!("{v} is not finite")));
    }
    let f: fn(&[f32], &mut Vec<u8>) = match ty.name {
        "q4_0" => |x, out| legacy(x, 8, false, out),
        "q4_1" => |x, out| legacy(x, 4, true, out),
        "q5_0" => |x, out| legacy(x, 16, false, out),
        "q5_1" => |x, out| legacy(x, 5, true, out),
        "q8_0" => q8_0,
        "q2_k" => q2_k,
        "q3_k" => q3_k,
        "q4_k" => |x, out| q45_k(x, false, out),
        "q5_k" => |x, out| q45_k(x, true, out),
        "q6_k" => q6_k,
        "mxfp4" => mxfp4,
        "nvfp4" => nvfp4,
        "q1_0" => q1_0,
        _ => q2_0,
    };
    let mut bytes = Vec::with_capacity(x.len() / ty.block as usize * ty.size as usize);
    x.chunks_exact(ty.block as usize)
        .for_each(|b| f(b, &mut bytes));
    Ok(vec![OwnedPart {
        role: "blocks".into(),
        dtype: Dtype::U8,
        shape,
        bytes,
    }])
}

// ------------------------------------------------------------------------------ ggml's scalars

/// `GGML_FP16_TO_FP32`, as ggml computes it in software.
fn fp16_to_fp32(h: u16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    let normalized = f32::from_bits((two_w >> 4) + (0xE0 << 23)) * f32::from_bits(0x0780_0000);
    let denormalized = f32::from_bits((two_w >> 17) | (126 << 23)) - 0.5;
    let bits = if two_w < (1 << 27) {
        denormalized.to_bits()
    } else {
        normalized.to_bits()
    };
    f32::from_bits(sign | bits)
}

/// `GGML_FP32_TO_FP16`, as ggml computes it in software.
fn fp32_to_fp16(f: f32) -> u16 {
    let mut base = (f.abs() * f32::from_bits(0x7780_0000)) * f32::from_bits(0x0880_0000);
    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x8000_0000;
    let bias = (shl1_w & 0xFF00_0000).max(0x7100_0000);
    base += f32::from_bits((bias >> 1) + 0x0780_0000);
    let bits = base.to_bits();
    let nonsign = ((bits >> 13) & 0x7C00) + (bits & 0x0FFF);
    ((sign >> 16)
        | if shl1_w > 0xFF00_0000 {
            0x7E00
        } else {
            nonsign
        }) as u16
}

fn e8m0_to_fp32_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000 << x
    } else {
        (x as u32 - 1) << 23
    })
}

fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let (exp, man) = (((x >> 3) & 0xF) as i32, (x & 7) as f32);
    let raw = if exp == 0 {
        man * pow2(-9)
    } else {
        (1.0 + man / 8.0) * pow2(exp - 7)
    };
    raw * 0.5
}

fn fp32_to_ue4m3(mut x: f32) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    x = x.min(448.0);
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127 + 7;
    if e <= 0 {
        let man = cvtt(x * 512.0 + 0.5).min(7);
        return if man < 1 { 0 } else { man as u8 };
    }
    if e >= 15 {
        return 0x7E;
    }
    let mut m = ((bits >> 20) & 7) + ((bits >> 19) & 1);
    if m > 7 {
        m = 0;
        e += 1;
        if e >= 15 {
            return 0x7E;
        }
    }
    ((e << 3) as u32 | m) as u8
}

/// 2^e for e in -126..=127, exactly.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// A C float-to-int cast as x86-64 performs it (`cvttss2si`): truncation, and `INT_MIN` for NaN
/// and out-of-range values; narrower C targets keep the low bits.
fn cvtt(f: f32) -> i32 {
    if (-2147483648.0..2147483648.0).contains(&f) {
        f as i32
    } else {
        i32::MIN
    }
}

fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    (val.to_bits() & 0x007F_FFFF) as i32 - 0x0040_0000
}

/// `floorf(log2f(x))` for x > 0: the floor of a correctly rounded log2, which glibc's `log2f` is
/// on every positive float (checked on all 2^31): the binade's exponent `e`, or `e + 1` where
/// log2 rounds up to it, `x` lying within half an f32 step below `2^(e+1)` in log space.
fn floor_log2f(x: f32) -> f32 {
    if x == f32::INFINITY {
        return x;
    }
    let b = x.to_bits();
    let e = if b >= 0x0080_0000 {
        (b >> 23) as i32 - 127
    } else {
        31 - b.leading_zeros() as i32 - 149
    };
    let k = e + 1;
    if k == 0 {
        return e as f32;
    }
    let a = k.unsigned_abs();
    let p = 31 - a.leading_zeros() as i32;
    let gap = if k > 0 && a.is_power_of_two() {
        p - 24
    } else {
        p - 23
    };
    let t = f64::from_bits(((gap - 1 + 1023) as u64) << 52) * std::f64::consts::LN_2;
    let threshold = 1.0 - t + t * t / 2.0 - t * t * t / 6.0;
    let xr = x as f64 * f64::from_bits(((1023 - k) as u64) << 52);
    if xr > threshold {
        k as f32
    } else {
        e as f32
    }
}

fn push_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&fp32_to_fp16(v).to_le_bytes());
}

fn best_index_mxfp4(x: f32, e: f32) -> u8 {
    let mut best = 0;
    let mut best_err = (K[0] as f32 * e - x).abs();
    for (i, k) in K.iter().enumerate().skip(1) {
        let err = (*k as f32 * e - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ------------------------------------------------------------------------------ 32-element types

/// q4_0 (`nb` 8) and q5_0 (16): symmetric; q4_1 (4) and q5_1 (5): min and max.
fn legacy(x: &[f32], nb: i32, with_min: bool, out: &mut Vec<u8>) {
    let five = nb == 16 || nb == 5;
    let (d, min);
    if with_min {
        let (mut mn, mut mx) = (f32::MAX, -f32::MAX);
        for &v in x {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        d = (mx - mn) / ((1 << nb) - 1) as f32;
        min = mn;
    } else {
        let (mut amax, mut mx) = (0f32, 0f32);
        for &v in x {
            if amax < v.abs() {
                amax = v.abs();
                mx = v;
            }
        }
        d = mx / -(nb as f32);
        min = 0.0;
    }
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    if with_min {
        push_f16(out, min);
    }
    let quant = |v: f32| -> u8 {
        if with_min {
            let x0 = (v - min) * id;
            if five {
                cvtt(x0 + 0.5) as u8
            } else {
                15.min(cvtt(x0 + 0.5) as i8 as i32) as u8
            }
        } else {
            let t = cvtt(v * id + (nb as f32 + 0.5)) as i8 as i32;
            (2 * nb - 1).min(t) as u8
        }
    };
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let (xi0, xi1) = (quant(x[j]), quant(x[16 + j]));
        if five {
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 as u32 & 0x10) >> 4) << j;
            qh |= ((xi1 as u32 & 0x10) >> 4) << (j + 16);
        } else {
            qs[j] = xi0 | (xi1 << 4);
        }
    }
    if five {
        out.extend_from_slice(&qh.to_le_bytes());
    }
    out.extend_from_slice(&qs);
}

fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    let mut amax = 0f32;
    for &v in x {
        amax = if amax > v.abs() { amax } else { v.abs() };
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    out.extend(x.iter().map(|&v| cvtt((v * id).round()) as u8));
}

fn mxfp4(x: &[f32], out: &mut Vec<u8>) {
    let amax = x
        .iter()
        .fold(0f32, |a, v| if a < v.abs() { v.abs() } else { a });
    let e = if amax > 0.0 {
        cvtt(floor_log2f(amax) - 2.0 + 127.0) as u8
    } else {
        0
    };
    let d = e8m0_to_fp32_half(e);
    out.push(e);
    out.extend((0..16).map(|j| best_index_mxfp4(x[j], d) | (best_index_mxfp4(x[16 + j], d) << 4)));
}

fn nvfp4(x: &[f32], out: &mut Vec<u8>) {
    let mut blk = [0u8; 36];
    for (s, xb) in x.as_chunks::<16>().0.iter().enumerate() {
        let amax = xb
            .iter()
            .fold(0f32, |a, v| if a < v.abs() { v.abs() } else { a });
        let ue = fp32_to_ue4m3(amax / 6.0);
        blk[s] = ue;
        let d = ue4m3_to_fp32(ue);
        for j in 0..8 {
            blk[4 + 8 * s + j] = best_index_mxfp4(xb[j], d) | (best_index_mxfp4(xb[8 + j], d) << 4);
        }
    }
    out.extend_from_slice(&blk);
}

fn q1_0(x: &[f32], out: &mut Vec<u8>) {
    let mut sum_abs = 0f32;
    for &v in x {
        sum_abs += v.abs();
    }
    push_f16(out, sum_abs / 128.0);
    let mut qs = [0u8; 16];
    for (j, &v) in x.iter().enumerate() {
        if v >= 0.0 {
            qs[j / 8] |= 1 << (j % 8);
        }
    }
    out.extend_from_slice(&qs);
}

fn q2_0(x: &[f32], out: &mut Vec<u8>) {
    let d = x
        .iter()
        .fold(0f32, |a, v| if v.abs() > a { v.abs() } else { a });
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    let mut qs = [0u8; 16];
    for (j, &w) in x.iter().enumerate() {
        let q = cvtt((w * id).round()).wrapping_add(1).clamp(0, 3);
        qs[j / 4] |= (q as u8) << ((j % 4) * 2);
    }
    out.extend_from_slice(&qs);
}

// ------------------------------------------------------------------------------ k-quant helpers

/// `make_qx_quants(n, nmax, x, L, rmse_type = 1, NULL)`, the only form the reference uses.
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
    let clamp = |v: i32| (-nmax).max((nmax - 1).min(v));
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l.fill(0);
        return 0.0;
    }
    let sums = |iscale: f32| {
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for &v in x {
            let li = clamp(nearest_int(iscale * v)) as f32;
            let w = v * v;
            sumlx += w * v * li;
            suml2 += w * li * li;
        }
        (sumlx, suml2)
    };
    let set = |l: &mut [i8], iscale: f32| {
        for (li, &v) in l.iter_mut().zip(x) {
            *li = (nmax + clamp(nearest_int(iscale * v))) as i8;
        }
    };
    let iscale = -(nmax as f32) / max;
    set(l, iscale);
    let (sumlx, suml2) = sums(iscale);
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in (-9..=9).filter(|&is| is != 0) {
        let iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let (sumlx, suml2) = sums(iscale);
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            set(l, iscale);
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// `make_q3_quants(n, nmax, x, L, do_rmse = true)`, the only form the reference uses.
fn make_q3_quants(nmax: i32, x: &[f32], l: &mut [i8]) -> f32 {
    let clamp = |v: i32| (-nmax).max((nmax - 1).min(v));
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l.fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for i in 0..x.len() {
        let li = clamp(nearest_int(iscale * x[i]));
        l[i] = li as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * li as f32 * li as f32;
    }
    for _ in 0..5 {
        let mut changed = 0;
        for i in 0..x.len() {
            let w = x[i] * x[i];
            let li = l[i] as f32;
            let mut slx = sumlx - w * x[i] * li;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * li * li;
                let new_l = clamp(nearest_int(x[i] * sl2 / slx));
                if new_l != l[i] as i32 {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * new_l as f32 * new_l as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        changed += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    for li in l.iter_mut() {
        *li = li.wrapping_add(nmax as i8);
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// `make_qkx2_quants` with `rdelta` 0.1 (every reference caller): `(scale, the_min)`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let err = |d: f32| if use_mad { d.abs() } else { d * d };
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        l[i] = 0.max(nmax.min(nearest_int(iscale * (x[i] - min)))) as u8;
        best_error += weights[i] * err(scale * l[i] as f32 + min - x[i]);
    }
    for is in 0..=nstep {
        iscale = (rmin + 0.1 * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let li = 0.max(nmax.min(nearest_int(iscale * (x[i] - min))));
            laux[i] = li as u8;
            let (w, lf) = (weights[i], li as f32);
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..n {
                cur_error += weights[i] * err(this_scale * laux[i] as f32 + this_min - x[i]);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// q2_k/q3_k's 2-bit codes: `qs[j/4 + l]` holds `L[j+l]`, `L[j+l+32]`, `L[j+l+64]`, `L[j+l+96]`.
fn pack2(l: impl Fn(usize) -> i32, qs: &mut [u8; 64]) {
    for j in (0..256).step_by(128) {
        for k in 0..32 {
            let v = l(j + k) | (l(j + k + 32) << 2) | (l(j + k + 64) << 4) | (l(j + k + 96) << 6);
            qs[j / 4 + k] = v as u8;
        }
    }
}

// ------------------------------------------------------------------------------ k-quants

fn q2_k(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0u8; 256];
    let mut laux = [0u8; 16];
    let mut weights = [0f32; 16];
    let (mut mins, mut scales) = ([0f32; 16], [0f32; 16]);
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..16 {
        let xs = &x[16 * j..16 * j + 16];
        for (w, v) in weights.iter_mut().zip(xs) {
            *w = v.abs();
        }
        let ls = &mut l[16 * j..16 * j + 16];
        let (s, m) = make_qkx2_quants(3, xs, &weights, ls, &mut laux, -0.5, 15, true);
        scales[j] = s;
        mins[j] = m;
        max_scale = if s > max_scale { s } else { max_scale };
        max_min = if m > max_min { m } else { max_min };
    }
    let mut sc = [0u8; 16];
    let d = if max_scale > 0.0 {
        let iscale = 15.0 / max_scale;
        for j in 0..16 {
            sc[j] = nearest_int(iscale * scales[j]) as u8;
        }
        fp32_to_fp16(max_scale / 15.0)
    } else {
        fp32_to_fp16(0.0)
    };
    let dmin = if max_min > 0.0 {
        let iscale = 15.0 / max_min;
        for j in 0..16 {
            sc[j] |= (nearest_int(iscale * mins[j]) << 4) as u8;
        }
        fp32_to_fp16(max_min / 15.0)
    } else {
        fp32_to_fp16(0.0)
    };
    for j in 0..16 {
        let dd = fp16_to_fp32(d) * (sc[j] & 0xF) as f32;
        if dd == 0.0 {
            continue;
        }
        let dm = fp16_to_fp32(dmin) * (sc[j] >> 4) as f32;
        for ii in 0..16 {
            l[16 * j + ii] = nearest_int((x[16 * j + ii] + dm) / dd).clamp(0, 3) as u8;
        }
    }
    let mut qs = [0u8; 64];
    pack2(|i| l[i] as i32, &mut qs);
    out.extend_from_slice(&sc);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
}

fn q3_k(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0i8; 256];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut amax) = (0f32, 0f32);
    for j in 0..16 {
        scales[j] = make_q3_quants(4, &x[16 * j..16 * j + 16], &mut l[16 * j..16 * j + 16]);
        if scales[j].abs() > amax {
            amax = scales[j].abs();
            max_scale = scales[j];
        }
    }
    let mut sc = [0u8; 12];
    let d = if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for j in 0..16 {
            let mut li = nearest_int(iscale * scales[j]) as i8;
            li = ((-32).max(31.min(li as i32)) + 32) as i8;
            if j < 8 {
                sc[j] = (li & 0xF) as u8;
            } else {
                sc[j - 8] |= ((li & 0xF) as u8) << 4;
            }
            li >>= 4;
            sc[j % 4 + 8] |= ((li as i32) << (2 * (j / 4))) as u8;
        }
        fp32_to_fp16(1.0 / iscale)
    } else {
        fp32_to_fp16(0.0)
    };
    for j in 0..16 {
        let lo = (if j < 8 { sc[j] & 0xF } else { sc[j - 8] >> 4 }) as i32;
        let hi = ((sc[8 + j % 4] >> (2 * (j / 4))) & 3) as i32;
        let dd = fp16_to_fp32(d) * ((lo | (hi << 4)) - 32) as i8 as f32;
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            l[16 * j + ii] = (nearest_int(x[16 * j + ii] / dd).clamp(-4, 3) + 4) as i8;
        }
    }
    let mut hmask = [0u8; 32];
    let (mut m, mut hm) = (0usize, 1u8);
    for v in l.iter_mut() {
        if *v > 3 {
            hmask[m] |= hm;
            *v -= 4;
        }
        m += 1;
        if m == 32 {
            m = 0;
            hm = hm.wrapping_shl(1);
        }
    }
    let mut qs = [0u8; 64];
    pack2(|i| l[i] as i32, &mut qs);
    out.extend_from_slice(&hmask);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&sc);
    out.extend_from_slice(&d.to_le_bytes());
}

fn q45_k(x: &[f32], five: bool, out: &mut Vec<u8>) {
    let nmax = if five { 31 } else { 15 };
    let (rmin, nstep) = if five { (-0.5, 15) } else { (-1.0, 20) };
    let mut l = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let (mut mins, mut scales) = ([0f32; 8], [0f32; 8]);
    let (mut max_scale, mut max_min) = (0f32, 0f32);
    for j in 0..8 {
        let xs = &x[32 * j..32 * j + 32];
        let mut sum_x2 = 0f32;
        for &v in xs {
            sum_x2 += v * v;
        }
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, v) in weights.iter_mut().zip(xs) {
            *w = av_x + v.abs();
        }
        let ls = &mut l[32 * j..32 * j + 32];
        let (s, m) = make_qkx2_quants(nmax, xs, &weights, ls, &mut laux, rmin, nstep, false);
        scales[j] = s;
        mins[j] = m;
        max_scale = if s > max_scale { s } else { max_scale };
        max_min = if m > max_min { m } else { max_min };
    }
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc = [0u8; 12];
    for j in 0..8 {
        let ls = 63.min(nearest_int(inv_scale * scales[j]) as u8);
        let lm = 63.min(nearest_int(inv_min * mins[j]) as u8);
        if j < 4 {
            sc[j] = ls;
            sc[j + 4] = lm;
        } else {
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc[j - 4] |= (ls >> 4) << 6;
            sc[j] |= (lm >> 4) << 6;
        }
    }
    let d = fp32_to_fp16(max_scale / 63.0);
    let dmin = fp32_to_fp16(max_min / 63.0);
    for j in 0..8 {
        let (s, m) = scale_min_k4(j, &sc);
        let dd = fp16_to_fp32(d) * s as f32;
        if dd == 0.0 {
            continue;
        }
        let dm = fp16_to_fp32(dmin) * m as f32;
        for ii in 0..32 {
            l[32 * j + ii] = nearest_int((x[32 * j + ii] + dm) / dd).clamp(0, nmax) as u8;
        }
    }
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&sc);
    let mut qs = [0u8; 128];
    if five {
        let mut qh = [0u8; 32];
        let (mut m1, mut m2) = (1u8, 2u8);
        for n in 0..4 {
            for j in 0..32 {
                let (mut l1, mut l2) = (l[64 * n + j], l[64 * n + j + 32]);
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                qs[32 * n + j] = l1 | (l2 << 4);
            }
            m1 = m1.wrapping_shl(2);
            m2 = m2.wrapping_shl(2);
        }
        out.extend_from_slice(&qh);
    } else {
        for n in 0..4 {
            for k in 0..32 {
                qs[32 * n + k] = l[64 * n + k] | (l[64 * n + k + 32] << 4);
            }
        }
    }
    out.extend_from_slice(&qs);
}

fn q6_k(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0i8; 256];
    let mut scales = [0f32; 16];
    let (mut max_scale, mut max_abs_scale) = (0f32, 0f32);
    for ib in 0..16 {
        let scale = make_qx_quants(32, &x[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = scale;
        if scale.abs() > max_abs_scale {
            max_abs_scale = scale.abs();
            max_scale = scale;
        }
    }
    if max_abs_scale < 1e-15 {
        out.extend_from_slice(&[0u8; 210]);
        return;
    }
    let iscale = -128.0 / max_scale;
    let d = fp32_to_fp16(1.0 / iscale);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = 127.min(nearest_int(iscale * scales[ib])) as i8;
    }
    for j in 0..16 {
        let dd = fp16_to_fp32(d) * sc[j] as f32;
        if dd == 0.0 {
            continue;
        }
        for ii in 0..16 {
            l[16 * j + ii] = (nearest_int(x[16 * j + ii] / dd).clamp(-32, 31) + 32) as i8;
        }
    }
    let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
    for n in 0..2 {
        for k in 0..32 {
            let at = |o: usize| l[128 * n + k + o] as i32;
            ql[64 * n + k] = ((at(0) & 0xF) | ((at(64) & 0xF) << 4)) as u8;
            ql[64 * n + k + 32] = ((at(32) & 0xF) | ((at(96) & 0xF) << 4)) as u8;
            qh[32 * n + k] =
                ((at(0) >> 4) | ((at(32) >> 4) << 2) | ((at(64) >> 4) << 4) | ((at(96) >> 4) << 6))
                    as u8;
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(sc.iter().map(|&v| v as u8));
    out.extend_from_slice(&d.to_le_bytes());
}
