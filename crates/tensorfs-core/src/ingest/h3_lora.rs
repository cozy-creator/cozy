//! Reviewed trainer H3 LoRA dialects -> canonical projection factors.
//!
//! These adapters omit base Q/K norm tensors, so this dialect declares the exact
//! H3 geometry (56 heads of 128). Every target shape must agree before a plan is
//! emitted. Trainer QKV factors use contiguous Q/K/V thirds: the original base
//! checkpoint's head interleave is reordered before training, not exported again
//! in adapter state dictionaries. This is structural conversion, not permission
//! to apply a FL2VA-trained adapter to REF2VA. Runtime owns that binding and never
//! parses foreign keys.

use std::collections::BTreeMap;

use crate::{
    dtype::{checked_bytes, Dtype},
    err::{refuse, Code, Result},
    spec::EncodingSpec,
};

use super::{
    carrier::{SourceHeader, SourceTensor},
    convert::{Bytes, Effect, Op, Plan, RolePlan, Xform, H3_HEADS, H3_HEAD_DIM},
};

#[derive(Default)]
struct Pair<'a> {
    a: Option<&'a SourceTensor>,
    b: Option<&'a SourceTensor>,
    alpha: Option<&'a SourceTensor>,
}

/// Recognize only the two reviewed spellings. Never replace all underscores with
/// dots: projection names contain underscores and that loses target identity.
fn target(key: &str) -> Result<(String, &str)> {
    let (stem, role) = if let Some(k) = key.strip_suffix(".lora_A.weight") {
        (k, "a")
    } else if let Some(k) = key.strip_suffix(".lora_B.weight") {
        (k, "b")
    } else if let Some(k) = key.strip_suffix(".lora_down.weight") {
        (k, "a")
    } else if let Some(k) = key.strip_suffix(".lora_up.weight") {
        (k, "b")
    } else if let Some(k) = key.strip_suffix(".alpha") {
        (k, "alpha")
    } else {
        return refuse(
            Code::UNKNOWN_FIELD,
            format!("unrecognized H3 LoRA tensor {key}"),
        );
    };
    let native = if let Some(k) = stem.strip_prefix("diffusion_model.") {
        k.to_string()
    } else if let Some(k) = stem.strip_prefix("lora_unet_blocks_") {
        let Some((block, projection)) = k.split_once('_') else {
            return refuse(Code::KEY_GRAMMAR, format!("invalid H3 LoRA target {stem}"));
        };
        let leaf = match projection {
            "attn_qkv_proj" => "attn.qkv_proj",
            "attn_out_proj" => "attn.out_proj",
            "mlp_fc1" => "mlp.fc1",
            "mlp_fc2" => "mlp.fc2",
            _ => {
                return refuse(
                    Code::KEY_GRAMMAR,
                    format!("unknown H3 LoRA projection {stem}"),
                )
            }
        };
        format!("blocks.{block}.{leaf}")
    } else {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("unknown H3 LoRA key dialect {key}"),
        );
    };
    Ok((native, role))
}

fn projection(native: &str) -> Result<(String, &str, u64, u64)> {
    let (rest, prefix, depth) = if let Some(rest) = native.strip_prefix("token_refiner.blocks.") {
        (rest, "token_refiner.refiner_blocks", 2)
    } else if let Some(rest) = native.strip_prefix("blocks.") {
        (rest, "transformer_blocks", 50)
    } else {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("unsupported H3 LoRA target {native}"),
        );
    };
    let Some((block, leaf)) = rest.split_once('.') else {
        return refuse(Code::KEY_GRAMMAR, format!("invalid H3 block {native}"));
    };
    let index = block.parse::<usize>().ok();
    if index.is_none_or(|index| index >= depth || index.to_string() != block) {
        return refuse(
            Code::SHAPE_MISMATCH,
            format!("H3 block outside the reviewed geometry: {native}"),
        );
    }
    let (input, output) = match leaf {
        "attn.qkv_proj" => (5376, H3_HEADS * H3_HEAD_DIM * 3),
        "attn.out_proj" => (7168, 5376),
        "mlp.fc1" => (5376, 28672),
        "mlp.fc2" => (14336, 5376),
        _ => {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("unsupported H3 LoRA target {native}"),
            )
        }
    };
    Ok((format!("{prefix}.{block}"), leaf, input, output))
}

fn operation(
    component: &str,
    file: usize,
    plain: &EncodingSpec,
    tensor: &SourceTensor,
    key: String,
    shape: Vec<u64>,
    xform: Option<Xform>,
) -> Op {
    let effect = if xform.is_some() {
        Effect::Permute
    } else {
        Effect::Rekey
    };
    let bytes = match xform {
        Some(xform) => Bytes::Permute {
            file,
            key: tensor.key.clone(),
            xform,
        },
        None => Bytes::Stream {
            file,
            key: tensor.key.clone(),
        },
    };
    Op {
        component: component.to_string(),
        out_key: key,
        encoding: "plain/1".into(),
        encoding_id: plain.object_id(),
        logical_dtype: tensor.dtype,
        logical_shape: shape.clone(),
        roles: vec![RolePlan {
            role: "value".into(),
            dtype: tensor.dtype,
            shape,
            bytes,
        }],
        effect,
    }
}

pub(super) fn plan(
    component: &str,
    file: usize,
    header: &SourceHeader,
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    let mut pairs: BTreeMap<String, Pair<'_>> = BTreeMap::new();
    for tensor in &header.tensors {
        if !matches!(tensor.dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
            return refuse(
                Code::DTYPE_MISMATCH,
                format!("{}: H3 LoRA factors must be floating point", tensor.key),
            );
        }
        let (name, role) = target(&tensor.key)?;
        let pair = pairs.entry(name).or_default();
        let slot = match role {
            "a" => &mut pair.a,
            "b" => &mut pair.b,
            _ => &mut pair.alpha,
        };
        if slot.replace(tensor).is_some() {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("duplicate H3 LoRA factor for {}", tensor.key),
            );
        }
    }
    p.header_bytes += header.header_bytes;
    for (name, pair) in pairs {
        let (Some(a), Some(b)) = (pair.a, pair.b) else {
            return refuse(
                Code::MISSING_COMPANION_ROLE,
                format!("{name}: both LoRA A and B are required"),
            );
        };
        let (prefix, leaf, input, output) = projection(&name)?;
        if a.shape.len() != 2
            || b.shape.len() != 2
            || a.shape[0] == 0
            || a.shape[0] > input.min(output)
            || a.shape[1] != input
            || b.shape != [output, a.shape[0]]
            || a.dtype != b.dtype
        {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{name}: LoRA A/B must be [rank,{input}] and [{output},rank] in one dtype"),
            );
        }
        if pair.alpha.is_some_and(|alpha| !alpha.shape.is_empty()) {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{name}: alpha must be a scalar"),
            );
        }
        let destinations: &[&str] = match leaf {
            "attn.qkv_proj" => &["attn.to_q", "attn.to_k", "attn.to_v"],
            "attn.out_proj" => &["attn.to_out.0"],
            "mlp.fc1" => &["ff.net.0.proj"],
            "mlp.fc2" => &["ff.net.2"],
            _ => unreachable!(),
        };
        for (take, leaf_out) in destinations.iter().enumerate() {
            let base = format!("{prefix}.{leaf_out}");
            p.ops.push(operation(
                component,
                file,
                plain,
                a,
                format!("{base}.lora_A.weight"),
                a.shape.clone(),
                None,
            ));
            let mut shape = b.shape.clone();
            let transform = if leaf == "attn.qkv_proj" {
                shape[0] /= 3;
                Some(Xform::QkvSplit {
                    groups: 1,
                    shares: [1, 1, 1],
                    take: take as u8,
                    unit_bytes: checked_bytes(
                        &b.key,
                        &[H3_HEADS * H3_HEAD_DIM, a.shape[0]],
                        b.dtype,
                    )?,
                })
            } else if leaf == "mlp.fc1" {
                Some(Xform::SwapHalves {
                    half_bytes: b.nbytes() / 2,
                })
            } else {
                None
            };
            p.ops.push(operation(
                component,
                file,
                plain,
                b,
                format!("{base}.lora_B.weight"),
                shape,
                transform,
            ));
            if let Some(alpha) = pair.alpha {
                p.ops.push(operation(
                    component,
                    file,
                    plain,
                    alpha,
                    format!("{base}.alpha"),
                    vec![],
                    None,
                ));
            }
        }
    }
    Ok(())
}
