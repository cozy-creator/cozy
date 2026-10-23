//! Nunchaku's "v1 single-file" SVDQuant checkpoints (`quantization_config` `method: svdquant`).
//! A W4A4 linear's tensors are in its GEMM kernel's tiles (v1's `svdq_layout.py`, the inverse of
//! deepcompressor's packers): `qweight` `I8 [N, K/2]` in mma m16n8k64 fragments, `wscales`
//! `F8_E4M3 [K/16, N]` in the micro-scale split, `wcscales`, `smooth_factor` and `bias` in
//! `pack_scale`'s lanes, `proj_down` `[K, R]` and `proj_up` `[N, R]` in 16×16 low-rank packs
//! (or, cozy's `lowrank_quant`, plain int8/fp8 with per-32 scales); `wtscale` is one scalar.
//! An svdq-fp4 linear folds into `svdq.nvfp4` in natural layout, each part computed from those
//! tensors as it is read; a fused `to_qkv`, `add_qkv_proj` or `qkv_proj` splits into its three
//! projections. svdq-int4 and the AWQ W4A16 modulation layers (`wzeros`) have no scheme here and
//! are refused by name.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::{Declared, Folded, FoldedPart, Folds};
use crate::err::{refuse, Code, Result};
use crate::format::decode::elem::floats;
use crate::format::{Dtype, Encoding, GlobalScale, NibbleOrder, ScaleLayout};
use crate::import::source::{Derive, Group, Tensor};

/// A fused projection and the three it holds, in order along its rows (v1's `_FUSED_SPLITS`).
const FUSED: [(&str, [&str; 3]); 3] = [
    ("to_qkv", ["to_q", "to_k", "to_v"]),
    ("add_qkv_proj", ["add_q_proj", "add_k_proj", "add_v_proj"]),
    ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
];

/// A W4A4 linear's leaves. `smooth_factor_orig` (the smoothing before a LoRA was fused into the
/// branch) never enters its forward, so it is dropped.
const LEAVES: [&str; 11] = [
    "qweight", "wscales", "wtscale", "wcscales", "smooth_factor", "smooth_factor_orig", "proj_down",
    "proj_up", "proj_down_scale", "proj_up_scale", "bias",
];
const FLOATS: [Dtype; 3] = [Dtype::F32, Dtype::BF16, Dtype::F16];

/// `first` then each `(base, digit)`: a mixed-radix index.
fn mixed(first: usize, digits: &[(usize, usize)]) -> usize {
    digits.iter().fold(first, |at, (base, d)| at * base + d)
}

/// `qweight`'s nibble (low first) holding code `(i, j)`: `unpack_qweight`'s tile
/// `(n/128, k/64, 1, 8, 8, 4, 2, 2, 1, 8)` of `(n/128, ⌊n/16⌋%8, ⌊n/8⌋%2, n%8)`, `(k/64, ⌊k/32⌋%2, ⌊k/8⌋%4, k%8)`.
fn code_at(i: usize, j: usize, k: usize) -> usize {
    let d = [(8, i / 16 % 8), (8, i % 8), (4, j / 8 % 4), (2, i / 8 % 2), (2, j / 32 % 2), (8, j % 8)];
    mixed(i / 128 * (k / 64) + j / 64, &d)
}

/// `wscales`' element for row `i`, block `g` of `w` (`unpack_wscales`, the `(4, 4, 8)` row split).
fn scale_at(i: usize, g: usize, w: usize) -> usize {
    mixed(i / 128 * (w / 4) + g / 4, &[(8, i % 8), (4, i / 8 % 4), (4, i / 32 % 4), (4, g % 4)])
}

/// A `pack_scale` vector's element `i` (`unpack_vector`).
fn vector_at(i: usize) -> usize {
    mixed(i / 128, &[(8, i / 16 % 8), (4, i / 2 % 4), (2, i / 8 % 2), (2, i % 2)])
}

/// A packed low-rank factor's element at pack `(c, r)` of `rp` rank packs, place `(u, v)` in it
/// (`unpack_lowrank`): `proj_down [K, R]`'s `(K/16, R/16)` packs, `u` along R and `v` along K;
/// `proj_up [N, R]`'s `(N/16, R/16)`, `u` along N and `v` along R.
fn lowrank_at(c: usize, r: usize, rp: usize, u: usize, v: usize) -> usize {
    mixed(c * rp + r, &[(8, u % 8), (4, v / 2 % 4), (2, u / 8), (2, v / 8), (2, v % 2)])
}

/// Elements `width` bytes wide, gathered: element `e` of the output is source element `at(e)`.
fn gather(src: &[u8], count: usize, width: usize, at: impl Fn(usize) -> usize) -> Vec<u8> {
    (0..count).flat_map(|e| &src[at(e) * width..(at(e) + 1) * width]).copied().collect()
}

fn f32s(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

/// A part computed from `t`'s bytes (and the ranges `with` of its file).
fn derived(role: &str, dtype: Dtype, shape: Vec<u64>, t: &Tensor, with: Vec<(u64, u64)>,
    f: impl Fn(&[&[u8]]) -> Vec<u8> + Send + Sync + 'static) -> FoldedPart {
    let len = dtype.byte_len(&shape).expect("whole bytes");
    let derive = Some(Derive { with, len, f: Arc::new(f) });
    FoldedPart { role: role.into(), dtype, shape, at: Tensor { derive, ..t.clone() }, inline: None }
}

/// A part of these bytes.
fn constant(role: &str, shape: Vec<u64>, t: &Tensor, bytes: Vec<u8>) -> FoldedPart {
    let at = Tensor { len: bytes.len() as u64, encoded: None, derive: None, ..t.clone() };
    FoldedPart { role: role.into(), dtype: Dtype::F32, shape, at, inline: Some(bytes) }
}

fn refused<T>(what: &str, detail: impl std::fmt::Display) -> Result<T> {
    refuse(Code::UNREGISTERED_ENCODING, format!("{what}: {detail}"))
}

fn bad<T>(what: &str, leaf: &str, detail: impl std::fmt::Display) -> Result<T> {
    refuse(Code::CARRIER_GEOMETRY, format!("{what}.{leaf}: {detail}"))
}

/// Every svdq-fp4 linear in `group`, folded under its projections' `.weight` and `.bias`.
pub(super) fn fold_group(group: &Group, d: &Declared, logical: &dyn Fn(&BTreeSet<String>) -> Result<Dtype>) -> Result<Folds> {
    let label = group.label();
    let precision = d.configs.iter().find_map(|(_, doc)| doc["weight"]["dtype"].as_str()).unwrap_or_default();
    if !precision.contains("fp4") {
        let why = match precision.contains("int4") {
            true => "svdq-int4 (int4 W4A4, which no lane serves); import the model's svdq-fp4 file".to_string(),
            false => format!("svdquant with weight dtype {precision:?}, not fp4"),
        };
        return refused(&label, format!("Nunchaku {why}"));
    }
    let modules: Vec<&str> = group.tensors.keys().filter_map(|k| k.strip_suffix(".qweight")).collect();
    let held = |m: &str| -> Vec<String> {
        let leaf = |k: &&String| k.strip_prefix(m).and_then(|l| l.strip_prefix('.')).is_some_and(|l| !l.contains('.'));
        group.tensors.keys().filter(leaf).cloned().collect()
    };
    let consumed: BTreeSet<String> = modules.iter().flat_map(|m| held(m)).collect();
    let dtype = logical(&consumed)?;
    let mut folds = Folds::new();
    for m in modules {
        linear(group, m, held(m), &dtype, &mut folds)?;
    }
    Ok(folds)
}

/// One W4A4 linear `m` (its keys `keys`): one `svdq.nvfp4` weight (and plain bias) per projection.
fn linear(group: &Group, m: &str, keys: Vec<String>, logical: &Dtype, folds: &mut Folds) -> Result<()> {
    let what = format!("{} {m}", group.label());
    let at = |leaf: &str| group.tensors.get(&format!("{m}.{leaf}"));
    if at("wzeros").is_some() {
        return refused(&what, "a Nunchaku AWQ W4A16 layer (qweight, wscales, wzeros): no cozytensors scheme holds \
            its float zeros and its exporter's adaLN row interleave");
    }
    let other: Vec<&str> = keys.iter().map(|k| &k[m.len() + 1..]).filter(|l| !LEAVES.contains(l)).collect();
    if !other.is_empty() {
        return refused(&what, format!("svdq linear tensors this import does not read: {other:?}"));
    }
    let q = at("qweight").expect("a module of its qweight");
    let (n, k) = match q.shape[..] {
        [n, half] if matches!(q.dtype, Dtype::I8 | Dtype::U8) && n.is_multiple_of(128) && half.is_multiple_of(64) => {
            (n as usize, 2 * half as usize)
        }
        _ => return bad(&what, "qweight", format!("{} {:?}, not I8 [N, K/2] in whole 128×128 tiles", q.dtype, q.shape)),
    };
    let w = k / 16;
    let numel = |t: &Tensor| t.shape.iter().product::<u64>() as usize;
    let Some(s) = at("wscales").filter(|s| s.dtype == Dtype::F8_E4M3 && numel(s) == n * w) else {
        return bad(&what, "wscales", format!("{:?}, not F8_E4M3 [{w}, {n}]", at("wscales").map(|s| (&s.dtype, &s.shape))));
    };
    let float = |leaf: &str, count: usize| -> Result<Option<&Tensor>> {
        match at(leaf) {
            Some(t) if !FLOATS.contains(&t.dtype) || numel(t) != count => {
                bad(&what, leaf, format!("{} {:?}, not {count} floats", t.dtype, t.shape))
            }
            found => Ok(found),
        }
    };
    let (wt, wc, smooth, bias) = (float("wtscale", 1)?, float("wcscales", n)?, float("smooth_factor", k)?, float("bias", n)?);
    let (Some(down), Some(up)) = (at("proj_down"), at("proj_up")) else {
        return refused(&what, "an svdq linear without its low-rank branch (proj_down, proj_up)");
    };
    let r = down.shape.get(1).copied().unwrap_or(0) as usize;
    let quantized = matches!(down.dtype, Dtype::I8 | Dtype::F8_E4M3);
    let fits = down.shape == [k as u64, r as u64] && up.shape == [n as u64, r as u64] && up.dtype == down.dtype && r > 0
        && match quantized {
            true => k.is_multiple_of(32) && r.is_multiple_of(32),
            false => FLOATS.contains(&down.dtype) && r.is_multiple_of(16),
        };
    if !fits {
        let (d, u) = ((&down.dtype, &down.shape), (&up.dtype, &up.shape));
        return bad(&what, "proj_down", format!("{d:?} beside proj_up {u:?}: not [K, R] and [N, R] of one dtype, R in whole packs"));
    }
    let scales = match quantized {
        true => {
            let scale = |leaf: &str, count: usize| float(leaf, count)?.filter(|t| t.file == down.file).map_or_else(
                || bad(&what, leaf, "missing beside its quantized factor"), |t| Ok((t.start, t.len, t.dtype.clone())));
            Some((scale("proj_down_scale", k / 32 * r)?, scale("proj_up_scale", n * (r / 32))?))
        }
        false => None,
    };
    // The projections it holds: itself, or a fused one's three in equal row thirds.
    let (parent, leaf) = m.rsplit_once('.').map_or(("", m), |(p, l)| (p, l));
    let outs: Vec<String> = match FUSED.iter().find(|(fused, _)| *fused == leaf) {
        Some((_, three)) if n.is_multiple_of(3) => {
            three.iter().map(|o| if parent.is_empty() { o.to_string() } else { format!("{parent}.{o}") }).collect()
        }
        Some(_) => return bad(&what, "qweight", format!("{n} rows do not split into three projections")),
        None => vec![m.to_owned()],
    };
    let rows = n / outs.len();
    let (k64, r64) = (k as u64, r as u64);
    for (o, out) in outs.iter().enumerate() {
        let (r0, h) = (o * rows, rows as u64);
        let mut parts = vec![
            derived("weight", Dtype::F4, vec![h, k64], q, vec![], move |src| {
                let nib = |at: usize| (src[0][at / 2] >> (4 * (at % 2))) & 15;
                (r0..r0 + rows).flat_map(|i| (0..k).step_by(2).map(move |j| (i, j)))
                    .map(|(i, j)| nib(code_at(i, j, k)) | nib(code_at(i, j + 1, k)) << 4).collect()
            }),
            derived("scale", Dtype::F8_E4M3, vec![h, w as u64], s, vec![], move |src| {
                gather(src[0], rows * w, 1, |e| scale_at(r0 + e / w, e % w, w))
            }),
            match wt {
                Some(t) => {
                    let dtype = t.dtype.clone();
                    derived("scale_2", Dtype::F32, vec![], t, vec![], move |src| f32s(floats(&dtype, src[0]).into_iter()))
                }
                None => constant("scale_2", vec![], q, 1f32.to_le_bytes().to_vec()),
            },
            match smooth {
                Some(t) => {
                    let dtype = t.dtype.clone();
                    derived("pre_quant_scale", Dtype::F32, vec![k64], t, vec![], move |src| {
                        let v = floats(&dtype, src[0]);
                        f32s((0..k).map(|j| 1.0 / v[vector_at(j)]))
                    })
                }
                None => constant("pre_quant_scale", vec![k64], q, f32s((0..k).map(|_| 1.0))),
            },
        ];
        if let Some(t) = wc {
            let dtype = t.dtype.clone();
            parts.push(derived("channel_scale", Dtype::F32, vec![h], t, vec![], move |src| {
                let v = floats(&dtype, src[0]);
                f32s((r0..r0 + rows).map(|i| v[vector_at(i)]))
            }));
        }
        match &scales {
            None => {
                let (dtype, width) = (down.dtype.clone(), down.dtype.byte_len(&[1]).expect("a float") as usize);
                let rp = r / 16;
                parts.push(derived("lowrank_down", dtype.clone(), vec![r64, k64], down, vec![], move |src| {
                    gather(src[0], r * k, width, |e| lowrank_at(e % k / 16, e / k / 16, rp, e / k % 16, e % 16))
                }));
                parts.push(derived("lowrank_up", dtype, vec![h, r64], up, vec![], move |src| {
                    gather(src[0], rows * r, width, |e| lowrank_at((r0 + e / r) / 16, e % r / 16, rp, (r0 + e / r) % 16, e % 16))
                }));
            }
            // cozy's lowrank_quant: plain row-major codes and per-32 scales along each factor's
            // contraction (down's K, up's R), dequantized exactly in f32.
            Some(((ds, dl, dd), (us, ul, ud))) => {
                let (qd, dd, ud) = (down.dtype.clone(), dd.clone(), ud.clone());
                let qu = qd.clone();
                parts.push(derived("lowrank_down", Dtype::F32, vec![r64, k64], down, vec![(*ds, *dl)], move |src| {
                    let (q, s) = (floats(&qd, src[0]), floats(&dd, src[1]));
                    f32s((0..r * k).map(|e| { let (t, j) = (e / k, e % k); q[j * r + t] * s[j / 32 * r + t] }))
                }));
                parts.push(derived("lowrank_up", Dtype::F32, vec![h, r64], up, vec![(*us, *ul)], move |src| {
                    let (q, s) = (floats(&qu, src[0]), floats(&ud, src[1]));
                    f32s((0..rows * r).map(|e| { let (i, t) = (r0 + e / r, e % r); q[i * r + t] * s[i * (r / 32) + t / 32] }))
                }));
            }
        }
        let encoding = Encoding { scheme: "svdq.nvfp4".into(),
            ..Encoding::nvfp4(GlobalScale::Multiply, NibbleOrder::LowFirst, ScaleLayout::RowMajor) };
        let shape = vec![h, k64];
        let folded = Folded { encoding, shape: shape.clone(), parts };
        let weight = Tensor { dtype: logical.clone(), shape, encoded: Some(Box::new(folded)), derive: None, ..q.clone() };
        folds.insert(format!("{out}.weight"), (keys.clone(), weight));
        if let Some(b) = bias {
            let width = b.dtype.byte_len(&[1]).expect("a float") as usize;
            let derive = Derive { with: vec![], len: h * width as u64, f: Arc::new(move |src: &[&[u8]]| {
                gather(src[0], rows, width, |e| vector_at(r0 + e))
            }) };
            let bias = Tensor { shape: vec![h], derive: Some(derive), ..b.clone() };
            folds.insert(format!("{out}.bias"), (vec![], bias));
        }
    }
    Ok(())
}
