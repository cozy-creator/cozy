//! bitsandbytes carriers (spec §6 `bnb.*`). A 4-bit weight `<w>` (`U8 [⌈n/2⌉, 1]`) sits beside
//! `<w>.absmax`, `<w>.quant_map` (with double quant `<w>.nested_absmax` and
//! `<w>.nested_quant_map`) and `<w>.quant_state.bitsandbytes__<nf4|fp4>`, a `U8` JSON document
//! naming the scheme, the block size, the logical dtype and shape and the nested offset. An
//! int8 weight `<m>.weight` (`I8 [N, K]`) sits beside `<m>.SCB` and `<m>.weight_format`. These
//! documents declare each module, so a file with no config reads too (Forge's single-file nf4).

use std::collections::BTreeSet;

use serde::Deserialize;

use crate::err::{refuse, Code, Result};
use crate::format::decode::bnb::{FP4_MAP, NF4_MAP};
use crate::format::{Dtype, Encoding, Value};

use super::carriers::{bytes, Folded, FoldedPart, Folds};
use super::source::{Group, SourceSet, Tensor};

const STATE: &str = ".quant_state.bitsandbytes__";

/// bnb's `quant_state` document; any other key is refused.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantState {
    quant_type: String,
    blocksize: u64,
    dtype: String,
    shape: Vec<u64>,
    nested_blocksize: Option<u64>,
    nested_dtype: Option<String>,
    nested_offset: Option<f64>,
}

/// The group's own bnb declaration: a `quant_state` document, or an int8 module's
/// `weight_format` beside its `SCB`.
pub(super) fn declaration(group: &Group) -> Option<String> {
    let found = group.tensors.keys().find(|k| {
        k.contains(STATE)
            || k.strip_suffix(".weight_format")
                .is_some_and(|m| group.tensors.contains_key(&format!("{m}.SCB")))
    })?;
    Some(format!("{} {found}", group.label()))
}

/// Every bnb weight in `group`, under `<module>.weight`; `logical` gives int8's logical dtype
/// (the model's), since bnb records it only for 4-bit.
pub(super) fn fold_group(
    set: &SourceSet,
    group: &Group,
    logical: &dyn Fn(&BTreeSet<String>) -> Result<Dtype>,
) -> Result<Folds> {
    let mut folds = Folds::new();
    for (key, t) in &group.tensors {
        if let Some((w, kind)) = key.split_once(STATE) {
            let (consumed, tensor) = four_bit(set, group, w, kind, key, t)?;
            folds.insert(w.to_owned(), (consumed, tensor));
        }
    }
    let mut int8 = Vec::new();
    for key in group.tensors.keys() {
        if let Some(m) = key.strip_suffix(".SCB") {
            int8.push(eight_bit(set, group, m)?);
        }
    }
    if !int8.is_empty() {
        let consumed: BTreeSet<String> = int8.iter().flat_map(|(_, c, _)| c.clone()).collect();
        let dtype = logical(&consumed)?;
        for (w, consumed, mut tensor) in int8 {
            tensor.dtype = dtype.clone();
            folds.insert(w, (consumed, tensor));
        }
    }
    for (w, (consumed, _)) in &folds {
        let prefix = format!("{w}.");
        let unread: Vec<&String> = (group.tensors.range(prefix.clone()..))
            .map(|(k, _)| k)
            .take_while(|k| k.starts_with(&prefix))
            .filter(|k| !consumed.contains(k))
            .collect();
        if !unread.is_empty() {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{}: {unread:?} beside bitsandbytes {w}: no bnb part",
                    group.label()
                ),
            );
        }
    }
    Ok(folds)
}

fn fail<T>(code: Code, what: &str, detail: impl std::fmt::Display) -> Result<T> {
    refuse(code, format!("{what}: {detail}"))
}

fn part(role: &str, t: &Tensor) -> FoldedPart {
    FoldedPart {
        role: role.into(),
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
        at: t.clone(),
        inline: None,
    }
}

/// `role` of `<w>`, `dtype` and `shape` as the scheme states them.
fn companion<'g>(
    group: &'g Group,
    w: &str,
    role: &str,
    dtype: Dtype,
    shape: &[u64],
) -> Result<(String, &'g Tensor)> {
    let key = format!("{w}.{role}");
    let Some(t) = group.tensors.get(&key) else {
        return fail(Code::MISSING_COMPANION_ROLE, &key, "missing");
    };
    if t.dtype != dtype || t.shape != shape {
        return fail(
            Code::CARRIER_GEOMETRY,
            &key,
            format!("{} {:?}, not {dtype} {shape:?}", t.dtype, t.shape),
        );
    }
    Ok((key, t))
}

fn four_bit(
    set: &SourceSet,
    group: &Group,
    w: &str,
    kind: &str,
    key: &str,
    blob: &Tensor,
) -> Result<(Vec<String>, Tensor)> {
    let raw = bytes(set, key, blob)?;
    let qs: QuantState =
        serde_json::from_slice(&raw).or_else(|e| fail(Code::MALFORMED_JSON, key, e))?;
    let map = match (kind, qs.quant_type.as_str()) {
        ("nf4", "nf4") => &NF4_MAP,
        ("fp4", "fp4") => &FP4_MAP,
        _ => {
            return fail(
                Code::UNREGISTERED_ENCODING,
                key,
                format!("quant_type {:?}", qs.quant_type),
            )
        }
    };
    let dtype = match qs.dtype.as_str() {
        "bfloat16" => Dtype::BF16,
        "float16" => Dtype::F16,
        "float32" => Dtype::F32,
        other => return fail(Code::DTYPE_UNKNOWN, key, format!("dtype {other:?}")),
    };
    let dq =
        match (&qs.nested_blocksize, &qs.nested_dtype, qs.nested_offset) {
            (None, None, None) => false,
            (Some(256), Some(d), Some(o)) if d == "float32" && o.is_finite() => true,
            _ => return fail(
                Code::UNREGISTERED_ENCODING,
                key,
                "nested_blocksize 256, nested_dtype float32 and a nested_offset, or none of them",
            ),
        };
    if !matches!(qs.blocksize, 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096) {
        return fail(
            Code::UNREGISTERED_ENCODING,
            key,
            format!("blocksize {}", qs.blocksize),
        );
    }
    let n: u64 = qs.shape.iter().product();
    let nb = n.div_ceil(qs.blocksize);
    let (wkey, weight) = match group.tensors.get_key_value(w) {
        Some((k, t)) if t.dtype == Dtype::U8 && t.shape == [n.div_ceil(2), 1] => (k.clone(), t),
        Some((_, t)) => {
            return fail(
                Code::CARRIER_GEOMETRY,
                w,
                format!(
                    "{} {:?}, not U8 [{}, 1] (bnb's uint8 storage)",
                    t.dtype,
                    t.shape,
                    n.div_ceil(2)
                ),
            )
        }
        None => return fail(Code::MISSING_COMPANION_ROLE, w, "missing"),
    };
    let absmax_dtype = if dq { Dtype::U8 } else { Dtype::F32 };
    let mut roles = vec![
        ("absmax", absmax_dtype, vec![nb]),
        ("quant_map", Dtype::F32, vec![16]),
    ];
    if dq {
        roles.push(("nested_absmax", Dtype::F32, vec![nb.div_ceil(256)]));
        roles.push(("nested_quant_map", Dtype::F32, vec![256]));
    }
    let mut consumed = vec![key.to_owned(), wkey];
    let mut parts = vec![part("weight", weight)];
    for (role, dtype, shape) in roles {
        let (k, t) = companion(group, w, role, dtype, &shape)?;
        if role == "quant_map" {
            let got: Vec<f32> = (bytes(set, &k, t)?.as_chunks::<4>().0.iter())
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if got != map {
                return fail(
                    Code::UNREGISTERED_ENCODING,
                    &k,
                    format!("{got:?} is not bnb's {kind} table"),
                );
            }
        }
        parts.push(part(role, t));
        consumed.push(k);
    }
    if let Some(offset) = qs.nested_offset {
        parts.push(FoldedPart {
            role: "nested_offset".into(),
            dtype: Dtype::F32,
            shape: vec![],
            at: Tensor {
                dtype: Dtype::F32,
                shape: vec![],
                file: weight.file,
                start: 0,
                len: 4,
                encoded: None,
            },
            // bnb wrote the f32 offset as a JSON float, exactly.
            inline: Some((offset as f32).to_le_bytes().to_vec()),
        });
    }
    let encoding = Encoding {
        scheme: format!("bnb.{kind}"),
        params: [
            ("block_size".to_owned(), Value::from(qs.blocksize)),
            ("double_quant".to_owned(), Value::from(dq)),
        ]
        .into(),
    };
    let tensor = Tensor {
        dtype,
        shape: qs.shape,
        file: weight.file,
        start: weight.start,
        len: weight.len,
        encoded: Some(Box::new(Folded { encoding, parts })),
    };
    Ok((consumed, tensor))
}

fn eight_bit(set: &SourceSet, group: &Group, m: &str) -> Result<(String, Vec<String>, Tensor)> {
    let w = format!("{m}.weight");
    let Some(weight) = group.tensors.get(&w) else {
        return fail(Code::MISSING_COMPANION_ROLE, &w, "missing beside its SCB");
    };
    let &[rows, _] = weight.shape.as_slice() else {
        return fail(
            Code::CARRIER_GEOMETRY,
            &w,
            format!("{:?}, not [N, K]", weight.shape),
        );
    };
    if weight.dtype != Dtype::I8 {
        return fail(
            Code::CARRIER_GEOMETRY,
            &w,
            format!("{}, not I8", weight.dtype),
        );
    }
    let (scb, s) = companion(group, m, "SCB", Dtype::F32, &[rows])?;
    let mut consumed = vec![w.clone(), scb];
    let mut parts = vec![part("weight", weight), part("SCB", s)];
    if group.tensors.contains_key(&format!("{m}.weight_format")) {
        let (k, t) = companion(group, m, "weight_format", Dtype::U8, &[])?;
        if bytes(set, &k, t)? != [0] {
            return fail(
                Code::UNREGISTERED_ENCODING,
                &k,
                "not 0 (bnb's row-major layout)",
            );
        }
        parts.push(part("weight_format", t));
        consumed.push(k);
    }
    let encoding = Encoding {
        scheme: "bnb.int8".into(),
        params: Default::default(),
    };
    let tensor = Tensor {
        encoded: Some(Box::new(Folded { encoding, parts })),
        ..weight.clone()
    };
    Ok((w, consumed, tensor))
}
