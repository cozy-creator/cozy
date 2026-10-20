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

/// A LoRA key in PEFT's spelling, or `None` for a key that is no LoRA factor or alpha: kohya's
/// `<m>.lora_down[.weight]` and `.lora_up[.weight]` are `<m>.lora_A.weight` and `.lora_B.weight`,
/// PEFT's adapter name goes (`<m>.lora_A.default.weight`), and so does `base_model.model.`.
pub fn peft_key(key: &str) -> Option<String> {
    let key = key.strip_prefix(PEFT_PREFIX).unwrap_or(key);
    if let Some(m) = key.strip_suffix(ALPHA) {
        return Some(format!("{m}{ALPHA}"));
    }
    let weight = key.strip_suffix(".weight");
    for (from, to) in [(".lora_down", A), (".lora_up", B), (".lora_A", A), (".lora_B", B)] {
        if let Some(m) = weight.unwrap_or(key).strip_suffix(from) {
            return Some(format!("{m}{to}"));
        }
        // PEFT's adapter name: `<m>.lora_A.<name>.weight`.
        let named = weight.and_then(|w| w.rsplit_once('.'));
        if let Some(m) = named.and_then(|(m, name)| m.strip_suffix(from).filter(|_| !name.is_empty())) {
            return Some(format!("{m}{to}"));
        }
    }
    None
}

/// PEFT's `adapter_config.json` for LoRA `modules` (module, rank): `source`'s fields where it
/// states them (its patterns as they are: a caller that folds them into `.alpha` empties them),
/// else `r` the commonest rank, `target_modules` the modules, `lora_alpha` kohya's
/// `ss_network_alpha`, else `r` when every rank is `r` (omitted otherwise: each module's own
/// rank is its `α`), and `base_model_name_or_path` `base_model`, else the source's, else kohya's
/// `ss_sd_model_name`.
pub fn adapter_config(
    modules: impl IntoIterator<Item = (String, u64)>,
    metadata: &BTreeMap<String, String>,
    source: Option<&Config>,
    base_model: Option<&str>,
) -> Config {
    let modules: BTreeMap<String, u64> = modules.into_iter().collect();
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for rank in modules.values() {
        *counts.entry(*rank).or_default() += 1;
    }
    let r = (counts.iter().rev()).max_by_key(|(_, n)| **n).map_or(0, |(r, _)| *r);
    let mut config = source.cloned().unwrap_or_default();
    config.insert("peft_type".into(), "LORA".into());
    config.entry("r").or_insert(json!(r));
    let stated = metadata.get("ss_network_alpha").and_then(|a| a.trim().parse::<f64>().ok());
    let alpha = match stated {
        Some(a) if a.fract() == 0.0 => Some(json!(a as i64)),
        Some(a) => Some(json!(a)),
        None => (counts.len() == 1).then(|| json!(r)),
    };
    if let (false, Some(alpha)) = (config.contains_key("lora_alpha"), alpha) {
        config.insert("lora_alpha".into(), alpha);
    }
    config.entry("target_modules").or_insert(json!(modules.keys().collect::<Vec<_>>()));
    let base = (base_model.map(str::to_owned))
        .or_else(|| config.get("base_model_name_or_path").and_then(Json::as_str).map(str::to_owned))
        .or_else(|| metadata.get("ss_sd_model_name").cloned());
    if let Some(base) = base {
        config.insert("base_model_name_or_path".into(), json!(base));
    }
    config
}

/// One module's factors: its rank and where its `lora_B` is planned.
struct Module {
    rank: u64,
    b: usize,
}

/// A LoRA component's config, and whether its source stated one (PEFT's keys are then the base
/// model's module names).
pub struct Adapter {
    pub config: Config,
    pub stated: bool,
}

/// Normalizes, in `planned`, each component holding LoRA factors ([`peft_key`]), and returns its
/// adapter. `sources` is each component's source group; an adapter config file read becomes the
/// component's config and leaves `files`.
pub fn normalize(
    planned: &mut Vec<Planned>,
    sources: &BTreeMap<String, &Group>,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, Adapter>> {
    let (mut adapters, mut read, mut alphas) = (BTreeMap::new(), BTreeSet::new(), Vec::new());
    for (component, group) in sources {
        let mine: Vec<usize> = (0..planned.len())
            .filter(|i| planned[*i].component == *component)
            .collect();
        if !mine.iter().any(|i| peft_key(&planned[*i].key).is_some_and(|k| k.ends_with(A))) {
            continue;
        }
        let mut seen = BTreeSet::new();
        let (mut modules, mut kept) = (BTreeMap::new(), BTreeSet::new());
        for i in &mine {
            let p = &mut planned[*i];
            if let Some(key) = peft_key(&p.key) {
                p.key = key;
            }
            if !seen.insert(p.key.clone()) {
                return refuse(Code::DUPLICATE_KEY, format!("{component}: two LoRA keys are {}", p.key));
            }
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
        let source = match (files.get(&path), group.metadata.get(DIFFUSERS_CONFIG)) {
            (Some(bytes), _) => {
                read.insert(path.clone());
                Some(parse(&path, bytes)?)
            }
            (None, Some(text)) => Some(parse(DIFFUSERS_CONFIG, text.as_bytes())?),
            (None, None) => None,
        };
        let source = match source {
            Some(source) => Some(fold(component, source, &modules, &kept, &mut alphas)?),
            None => None,
        };
        let ranks = modules.iter().map(|(m, v)| (m.clone(), v.rank));
        let config = adapter_config(ranks, &group.metadata, source.as_ref(), None);
        let stated = source.is_some();
        adapters.insert(component.clone(), Adapter { config, stated });
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
    Ok(adapters)
}

fn parse(what: &str, bytes: &[u8]) -> Result<Config> {
    serde_json::from_slice(bytes).map_err(|e| Refusal {
        code: Code::MALFORMED_JSON,
        detail: format!("{what}: {e}"),
    })
}

/// A PEFT `LoraConfig` with `alpha_pattern` and `use_rslora` folded into each module's `.alpha`
/// (PEFT's scale is `α_m / r_m`, or `α_m / √r_m` with rsLoRA) and emptied, as is `rank_pattern`,
/// each `lora_A`'s rows already.
fn fold(
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
    Ok(config)
}
