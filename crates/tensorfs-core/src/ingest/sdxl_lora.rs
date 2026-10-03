//! Kohya SDXL LoRAs -> the PEFT factors cozy-runtime composes, one output component per
//! base component the factors adapt: `unet`, `text_encoder`, `text_encoder_2`.
//!
//! A kohya key is a module path with its dots written as underscores under `lora_unet_`,
//! `lora_te1_` or `lora_te2_`. It is never unmangled: the reviewed module names are mangled
//! and looked up. They come from the SDXL route table, so a factor lands on exactly the key
//! the SDXL single-file converter gives its base weight. UNet names are accepted in both
//! spellings trainers write, the original LDM one (`input_blocks_4_1_...`, kohya
//! sd-scripts) and the Diffusers one (`down_blocks_1_attentions_0_...`). Targets are the
//! linear layers of the UNet transformer blocks and of both CLIP towers. Conv (LoCon)
//! factors refuse: the runtime composes linear layers only. Bytes stream unchanged; the
//! runtime scales by alpha/rank * strength, as kohya does.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use crate::{
    dtype::Dtype,
    err::{refuse, Code, Result},
    spec::EncodingSpec,
};

use super::{
    carrier::{SourceHeader, SourceTensor},
    convert::Plan,
    h3_lora::operation,
    routes,
};

pub(super) const CONVERTER: &str = "sdxl.lora/1";

struct Module {
    component: &'static str,
    path: String,
    input: u64,
    output: u64,
}

/// `(input, output)` of a reviewed LoRA target in canonical (route-target) spelling.
fn geometry(component: &str, path: &str) -> Option<(u64, u64)> {
    let (hidden, inner, layer) = match component {
        "text_encoder" => (768, 3072, path.strip_prefix("encoder.layers.")?),
        "text_encoder_2" => (1280, 5120, path.strip_prefix("text_model.encoder.layers.")?),
        "unet" => return unet_geometry(path),
        _ => return None,
    };
    Some(match layer.split_once('.')?.1 {
        "self_attn.q_proj" | "self_attn.k_proj" | "self_attn.v_proj" | "self_attn.out_proj" => {
            (hidden, hidden)
        }
        "mlp.fc1" => (hidden, inner),
        "mlp.fc2" => (inner, hidden),
        _ => return None,
    })
}

fn unet_geometry(path: &str) -> Option<(u64, u64)> {
    let (channels, rest) = [
        ("down_blocks.1.", 640),
        ("down_blocks.2.", 1280),
        ("mid_block.", 1280),
        ("up_blocks.0.", 1280),
        ("up_blocks.1.", 640),
    ]
    .into_iter()
    .find_map(|(prefix, channels)| path.strip_prefix(prefix).map(|rest| (channels, rest)))?;
    let leaf = rest.strip_prefix("attentions.")?.split_once('.')?.1;
    if leaf == "proj_in" || leaf == "proj_out" {
        return Some((channels, channels));
    }
    Some(
        match leaf.strip_prefix("transformer_blocks.")?.split_once('.')?.1 {
            "attn1.to_q" | "attn1.to_k" | "attn1.to_v" | "attn1.to_out.0" | "attn2.to_q"
            | "attn2.to_out.0" => (channels, channels),
            "attn2.to_k" | "attn2.to_v" => (2048, channels),
            "ff.net.0.proj" => (channels, 8 * channels),
            "ff.net.2" => (4 * channels, channels),
            _ => return None,
        },
    )
}

/// Every accepted kohya stem and the module it names.
fn modules() -> &'static HashMap<String, Module> {
    static INDEX: OnceLock<HashMap<String, Module>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let table = routes::table(routes::SDXL).expect("the SDXL route table is built in");
        let mut index = HashMap::new();
        for component in &table.components {
            let (prefix, name) = match component.name.as_str() {
                "unet" => ("lora_unet_", "unet"),
                "text_encoder" => ("lora_te1_", "text_encoder"),
                "text_encoder_2" => ("lora_te2_", "text_encoder_2"),
                _ => continue,
            };
            for route in &component.routes {
                let Some(path) = route.target.strip_suffix(".weight") else {
                    continue;
                };
                let Some((input, output)) = geometry(name, path) else {
                    continue;
                };
                // Trainers saw the LDM UNet (route source) or Diffusers' (target), CLIP-L
                // under transformers' `text_model.` (source), bigG as its target.
                let source = route.source.strip_suffix(".weight");
                let spellings = match name {
                    "unet" => vec![Some(path), source],
                    "text_encoder" => vec![source],
                    _ => vec![Some(path)],
                };
                for spelling in spellings.into_iter().flatten() {
                    let module = Module {
                        component: name,
                        path: path.to_string(),
                        input,
                        output,
                    };
                    index.insert(format!("{prefix}{}", spelling.replace('.', "_")), module);
                }
            }
        }
        index
    })
}

#[derive(Default)]
struct Pair<'a> {
    a: Option<&'a SourceTensor>,
    b: Option<&'a SourceTensor>,
    alpha: Option<&'a SourceTensor>,
}

fn role(key: &str) -> Result<(&str, &'static str)> {
    for (suffix, role) in [
        (".lora_down.weight", "a"),
        (".lora_up.weight", "b"),
        (".lora_A.weight", "a"),
        (".lora_B.weight", "b"),
        (".alpha", "alpha"),
    ] {
        if let Some(stem) = key.strip_suffix(suffix) {
            return Ok((stem, role));
        }
    }
    refuse(
        Code::UNKNOWN_FIELD,
        format!(
            "unrecognized SDXL LoRA tensor {key}: {CONVERTER} maps LoRA down/up factors and \
             alpha (not LoHa, LoKr, DoRA or LoCon mid factors)"
        ),
    )
}

pub(super) fn plan(
    file: usize,
    header: &SourceHeader,
    plain: &EncodingSpec,
    p: &mut Plan,
) -> Result<()> {
    let conv: Vec<&str> = header
        .tensors
        .iter()
        .filter(|tensor| tensor.shape.len() == 4)
        .map(|tensor| tensor.key.as_str())
        .collect();
    if !conv.is_empty() {
        return refuse(
            Code::ADAPTER_TARGET_UNSUPPORTED,
            format!(
                "{} conv LoRA (LoCon) factor(s), first {}. The runtime composes linear \
                 layers only, and dropping conv factors changes the adapter. Store the file \
                 as-is.",
                conv.len(),
                conv[0],
            ),
        );
    }
    let index = modules();
    let mut pairs: BTreeMap<(&str, &str), (&Module, Pair<'_>)> = BTreeMap::new();
    for tensor in &header.tensors {
        if !matches!(tensor.dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) {
            return refuse(
                Code::DTYPE_MISMATCH,
                format!("{}: SDXL LoRA factors must be floating point", tensor.key),
            );
        }
        let (stem, role) = role(&tensor.key)?;
        let Some(module) = index.get(stem) else {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("{stem}: not a kohya SDXL LoRA target (UNet transformer or CLIP linear)"),
            );
        };
        let (_, pair) = pairs
            .entry((module.component, module.path.as_str()))
            .or_insert_with(|| (module, Pair::default()));
        let slot = match role {
            "a" => &mut pair.a,
            "b" => &mut pair.b,
            _ => &mut pair.alpha,
        };
        if slot.replace(tensor).is_some() {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("{}: a second {role} factor for {}", tensor.key, module.path),
            );
        }
    }
    p.header_bytes += header.header_bytes;
    for (module, pair) in pairs.into_values() {
        let name = format!("{}.{}", module.component, module.path);
        let (Some(a), Some(b)) = (pair.a, pair.b) else {
            return refuse(
                Code::MISSING_COMPANION_ROLE,
                format!("{name}: both LoRA down and up factors are required"),
            );
        };
        let (input, output) = (module.input, module.output);
        if a.shape.len() != 2
            || a.shape[0] == 0
            || a.shape[0] > input.min(output)
            || a.shape[1] != input
            || b.shape != [output, a.shape[0]]
            || a.dtype != b.dtype
        {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!(
                    "{name}: LoRA down/up must be [rank,{input}] and [{output},rank] in one dtype"
                ),
            );
        }
        if pair.alpha.is_some_and(|alpha| !alpha.shape.is_empty()) {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{name}: alpha must be a scalar"),
            );
        }
        let factors = [
            (".lora_A.weight", Some(a)),
            (".lora_B.weight", Some(b)),
            (".alpha", pair.alpha),
        ];
        for (suffix, tensor) in factors {
            if let Some(tensor) = tensor {
                p.ops.push(operation(
                    module.component,
                    file,
                    plain,
                    tensor,
                    format!("{}{suffix}", module.path),
                    tensor.shape.clone(),
                    None,
                ));
            }
        }
    }
    Ok(())
}
