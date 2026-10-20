//! LoRA components in the D10 form (spec §4): factors under the module names they adapt, as
//! PEFT's `<module>.lora_A.weight` `[r, in]` and `.lora_B.weight` `[out, r]`, applied as
//! `W' = W + (α / r) · (B · A)` with `r` `lora_A`'s rows and `α` the module's `<module>.alpha`,
//! else the config's `lora_alpha`, else `r`. The component's config is PEFT's
//! `adapter_config.json`: the source's own (its file, or diffusers' `lora_adapter_metadata`) with
//! `alpha_pattern`, `rank_pattern` and `use_rslora` folded into per-module `.alpha`, or kohya's
//! training metadata mapped. Kohya's `.alpha` tensors stay as they are.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value as Json};

use super::engine::{Planned, PlannedPart, Reading};
use super::source::Group;
use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{Config, Dtype};

/// PEFT's adapter config: a LoRA component's config file.
pub const ADAPTER_CONFIG: &str = "adapter_config.json";
const A: &str = ".lora_A.weight";
const B: &str = ".lora_B.weight";
const ALPHA: &str = ".alpha";
/// `get_peft_model_state_dict`'s wrapper around the base model's module names.
const PEFT_PREFIX: &str = "base_model.model.";
/// diffusers' `save_lora_adapter` records the `LoraConfig` here, as JSON.
const DIFFUSERS_CONFIG: &str = "lora_adapter_metadata";

/// One module's factors: its rank and where its `lora_B` is planned.
struct Module {
    rank: u64,
    b: usize,
}

/// Normalizes, in `planned`, each component holding LoRA factors, and returns its config.
/// `sources` is each component's source group; an adapter config file read becomes the
/// component's config and leaves `files`.
pub fn normalize(
    planned: &mut Vec<Planned>,
    sources: &BTreeMap<String, &Group>,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, Config>> {
    let (mut configs, mut read, mut alphas) = (BTreeMap::new(), BTreeSet::new(), Vec::new());
    for (component, group) in sources {
        let mine: Vec<usize> = (0..planned.len())
            .filter(|i| planned[*i].component == *component)
            .collect();
        if !mine.iter().any(|i| planned[*i].key.ends_with(A)) {
            continue;
        }
        if mine.iter().all(|i| planned[*i].key.starts_with(PEFT_PREFIX)) {
            for i in &mine {
                planned[*i].key.drain(..PEFT_PREFIX.len());
            }
        }
        let mut modules = BTreeMap::new();
        let mut kept = BTreeSet::new();
        for i in &mine {
            let p = &planned[*i];
            if let Some(m) = p.key.strip_suffix(A) {
                modules.entry(m.to_owned()).or_insert(Module { rank: 0, b: *i }).rank =
                    p.shape.first().copied().unwrap_or(0);
            } else if let Some(m) = p.key.strip_suffix(B) {
                modules.entry(m.to_owned()).or_insert(Module { rank: 0, b: *i }).b = *i;
            } else if let Some(m) = p.key.strip_suffix(ALPHA) {
                kept.insert(m.to_owned());
            }
        }
        let path = super::join(&group.folder, ADAPTER_CONFIG);
        let stated = match (files.get(&path), group.metadata.get(DIFFUSERS_CONFIG)) {
            (Some(bytes), _) => {
                read.insert(path.clone());
                Some(parse(&path, bytes)?)
            }
            (None, Some(text)) => Some(parse(DIFFUSERS_CONFIG, text.as_bytes())?),
            (None, None) => None,
        };
        let config = match stated {
            Some(config) => peft(component, config, &modules, &kept, &mut alphas)?,
            None => kohya(&group.metadata, &modules),
        };
        configs.insert(component.clone(), config);
    }
    files.retain(|path, _| !read.contains(path));
    // Each new `.alpha` follows its module's `lora_B`; from the back, so indices hold.
    alphas.sort_by_key(|(at, ..)| std::cmp::Reverse(*at));
    for (at, component, module, alpha) in alphas {
        let part = PlannedPart {
            role: String::new(),
            dtype: Dtype::F32,
            shape: vec![],
            read: Reading::inline(alpha.to_le_bytes().to_vec(), Dtype::F32),
        };
        let p = Planned {
            component,
            key: format!("{module}{ALPHA}"),
            dtype: Dtype::F32,
            shape: vec![],
            encoding: None,
            parts: vec![part],
        };
        planned.insert(at + 1, p);
    }
    Ok(configs)
}

fn parse(what: &str, bytes: &[u8]) -> Result<Config> {
    serde_json::from_slice(bytes).map_err(|e| Refusal {
        code: Code::MALFORMED_JSON,
        detail: format!("{what}: {e}"),
    })
}

/// A PEFT `LoraConfig`: `alpha_pattern` and `use_rslora` become each module's `.alpha` (PEFT's
/// scale is `α_m / r_m`, or `α_m / √r_m` with rsLoRA); `rank_pattern` is each `lora_A`'s rows
/// already. The config keeps everything else.
fn peft(
    component: &str,
    mut config: Config,
    modules: &BTreeMap<String, Module>,
    kept: &BTreeSet<String>,
    alphas: &mut Vec<(usize, String, String, f32)>,
) -> Result<Config> {
    let unsupported = |why: String| refuse(Code::ADAPTER_TARGET_UNSUPPORTED, format!("{component}: {why}"));
    match config.get("peft_type") {
        None | Some(Json::Null) => {}
        Some(t) if t == "LORA" => {}
        Some(t) => return unsupported(format!("peft_type {t} is not LORA")),
    }
    if config.get("use_dora") == Some(&Json::Bool(true)) {
        return unsupported("DoRA's magnitude vectors are not LoRA factors".into());
    }
    let number = |what: &str, v: &Json| {
        v.as_f64().ok_or_else(|| Refusal {
            code: Code::WRONG_TYPE,
            detail: format!("{component}: {what} {v} is not a number"),
        })
    };
    let alpha = match config.get("lora_alpha") {
        Some(v) => Some(number("lora_alpha", v)?),
        None => None,
    };
    let mut patterns = Vec::new();
    if let Some(map) = config.get("alpha_pattern").and_then(Json::as_object) {
        for (key, v) in map {
            if key.contains(['*', '+', '?', '[', ']', '(', ')', '{', '}', '|', '^', '$', '\\']) {
                return unsupported(format!("alpha_pattern key {key:?} is a regex, not a module name"));
            }
            patterns.push((key.clone(), number("alpha_pattern", v)?));
        }
    }
    let rslora = config.get("use_rslora") == Some(&Json::Bool(true));
    for (name, m) in modules.iter().filter(|(name, _)| !kept.contains(*name)) {
        let named: BTreeSet<u64> = (patterns.iter())
            .filter(|(key, _)| name == key || name.ends_with(&format!(".{key}")))
            .map(|(_, v)| v.to_bits())
            .collect();
        if named.len() > 1 {
            return unsupported(format!("alpha_pattern names {name} twice, differently"));
        }
        let named = named.first().map(|bits| f64::from_bits(*bits));
        let a = named.or(alpha).unwrap_or(m.rank as f64);
        if named.is_some() || rslora {
            let a = if rslora { a * (m.rank as f64).sqrt() } else { a };
            alphas.push((m.b, component.to_owned(), name.clone(), a as f32));
        }
    }
    for key in ["alpha_pattern", "rank_pattern"] {
        if config.contains_key(key) {
            config.insert(key.into(), json!({}));
        }
    }
    if rslora {
        config.insert("use_rslora".into(), Json::Bool(false));
    }
    config.insert("peft_type".into(), "LORA".into());
    Ok(config)
}

/// Kohya's training metadata as PEFT's config: `ss_network_dim` is `r` (else the commonest
/// rank), `ss_network_alpha` `lora_alpha` (else `r`), `ss_sd_model_name` the base model.
fn kohya(metadata: &BTreeMap<String, String>, modules: &BTreeMap<String, Module>) -> Config {
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for m in modules.values() {
        *counts.entry(m.rank).or_default() += 1;
    }
    let commonest = (counts.iter().rev()).max_by_key(|(_, n)| **n).map_or(0, |(r, _)| *r);
    let stated = |key: &str| metadata.get(key).and_then(|v| v.trim().parse::<f64>().ok());
    let r = stated("ss_network_dim").filter(|r| r.fract() == 0.0 && *r >= 1.0);
    let r = r.map_or(commonest, |r| r as u64);
    let alpha = match stated("ss_network_alpha") {
        Some(a) if a.fract() == 0.0 => json!(a as i64),
        Some(a) => json!(a),
        None => json!(r),
    };
    let mut config = Config::new();
    config.insert("peft_type".into(), "LORA".into());
    config.insert("r".into(), json!(r));
    config.insert("lora_alpha".into(), alpha);
    config.insert("target_modules".into(), json!(modules.keys().collect::<Vec<_>>()));
    if let Some(base) = metadata.get("ss_sd_model_name") {
        config.insert("base_model_name_or_path".into(), json!(base));
    }
    config
}
