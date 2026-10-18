//! Caller LoRA adapters as one view over a base checkpoint (the Python worker's
//! `lora_composition.prepare`): base `<path>.weight/.bias` become `<path>.base_layer.*`, factors
//! graft in as `<path>.lora_{A,B}.adapter_<i>.weight`, and `model_adapters.json` holds the
//! ordered graph the executor binds with PEFT. Nothing is computed, and large parts keep their
//! objects. Factors must already be canonical PEFT keys (`<path>.lora_A.weight`,
//! `<path>.lora_B.weight`, optional scalar `<path>.alpha`): foreign layouts are normalized when a
//! model is imported, never here. A view is pinned `view-<id>`, a cache that expires like the
//! machine's others (`storage::expire`).
use std::{collections::BTreeMap, io};

use serde::Serialize;
use tensorfs_core::format::{Body, Component, Dtype, Header, ObjectRef, Tensor};
use tensorfs_core::store2::{Checkpoint, Store};

use crate::storage::{self, pin};

pub const FORMAT: &str = "cozy.model.lora/1";
/// The graph's file in the view's manifest.
pub const GRAPH_FILE: &str = "model_adapters.json";

/// One caller adapter, in order. An empty `component` is inferred when exactly one base
/// component fits every factor pair; an empty `source_component` is the adapter's component
/// named like `component`, else `adapter`, else its only one.
#[derive(Clone, Debug)]
pub struct Selection {
    pub manifest: ObjectRef,
    pub component: String,
    pub source_component: String,
    pub strength: f64,
}

/// The composed checkpoint and the pin that keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Composed {
    pub pin: String,
    pub manifest: ObjectRef,
}

fn refuse(code: &str, detail: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, format!("adapter_{code}: {}", detail.into()))
}

#[derive(Serialize)]
struct Linear {
    component: String,
    target: String,
    adapter: String,
    rank: u64,
    alpha: f64,
    strength: f64,
    dtype: String,
}

#[derive(Serialize)]
struct AdapterRef {
    #[serde(rename = "ref")]
    reference: String,
    scale: f64,
    kind: &'static str,
    component: String,
    source_component: String,
    family: &'static str,
}

#[derive(Serialize)]
struct Graph {
    format: &'static str,
    layers: Vec<Linear>,
    adapters: Vec<AdapterRef>,
}

/// Every factor value finite (zero-strength inputs too); a scalar's value is returned.
fn validate(checkpoint: &Checkpoint, component: &str, key: &str, tensor: &Tensor) -> io::Result<Option<f64>> {
    let width = match tensor.dtype {
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 => 4,
        _ => 0,
    };
    if width == 0 || !matches!(tensor.body, Body::Data(_)) {
        return Err(refuse("tensor", format!("{component}.{key}: plain f16, bf16 or f32 factors required")));
    }
    let bytes = checkpoint.read(component, key).map_err(storage::io_error)?;
    let values: Vec<f64> = bytes
        .chunks_exact(width)
        .map(|b| match tensor.dtype {
            Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            Dtype::BF16 => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64,
            _ => half(u16::from_le_bytes([b[0], b[1]])),
        })
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(refuse("nonfinite", format!("{component}.{key}: factor contains NaN or infinity")));
    }
    Ok(if tensor.shape.is_empty() { values.first().copied() } else { None })
}

fn half(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f64;
    match exponent {
        0 => sign * fraction * 2f64.powi(-24),
        31 if fraction == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// Factor pairs by target path: role (`a`, `b`, `alpha`) → (key, tensor).
type Pairs<'h> = BTreeMap<String, BTreeMap<&'static str, (String, &'h Tensor)>>;

fn pairs(factors: &BTreeMap<String, Tensor>) -> io::Result<Pairs<'_>> {
    let mut pairs: Pairs = BTreeMap::new();
    for (key, tensor) in factors {
        let (path, role) = [(".lora_A.weight", "a"), (".lora_B.weight", "b"), (".alpha", "alpha")]
            .into_iter()
            .find_map(|(suffix, role)| key.strip_suffix(suffix).map(|path| (path, role)))
            .ok_or_else(|| refuse("key", format!("unsupported adapter tensor {key:?}")))?;
        if path.is_empty() {
            return Err(refuse("target", "an adapter target must name a child linear layer"));
        }
        pairs.entry(path.into()).or_default().insert(role, (key.clone(), tensor));
    }
    Ok(pairs)
}

/// A component's tensors by key.
fn keyed(component: &Component) -> BTreeMap<String, Tensor> {
    component.tensors.iter().map(|t| (t.key.clone(), t.clone())).collect()
}

fn infer_component(base: &Header, factors: &BTreeMap<String, Tensor>) -> io::Result<String> {
    let pairs = pairs(factors)?;
    let fits = |tensors: &BTreeMap<String, Tensor>| {
        !pairs.is_empty()
            && pairs.iter().all(|(path, pair)| match (pair.get("a"), pair.get("b"), tensors.get(&format!("{path}.weight"))) {
                (Some((_, a)), Some((_, b)), Some(weight)) => {
                    a.shape.len() == 2 && b.shape.len() == 2 && [b.shape[0], a.shape[1]] == weight.shape[..]
                }
                _ => false,
            })
    };
    let matches: Vec<&String> = base.components().iter().filter(|c| fits(&keyed(c))).map(|c| &c.name).collect();
    match matches.as_slice() {
        [one] => Ok((*one).clone()),
        _ => Err(refuse(
            "component",
            format!(
                "select one adapter component; compatible choices: {}",
                if matches.is_empty() { "none".into() } else { matches.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") }
            ),
        )),
    }
}

/// A tensor the view takes: `(component, key, source, source component, source key)`; source 0
/// is the base, `i + 1` adapter `i`.
type Take = (String, String, usize, String, String);

/// What the view takes, in execution order, and its graph, refusing unsupported keys before
/// anything is written.
fn plan(base: &Checkpoint, adapters: &[(Selection, Checkpoint)]) -> io::Result<(Vec<Take>, Graph)> {
    if base.manifest().files.contains_key(GRAPH_FILE) {
        return Err(refuse("nested", "select the original base and one complete ordered adapter stack"));
    }
    let header = base.header();
    let mut renamed: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut grafts: BTreeMap<String, Vec<Take>> = BTreeMap::new();
    let mut graph = Graph { format: FORMAT, layers: vec![], adapters: vec![] };
    for (index, (selection, adapter)) in adapters.iter().enumerate() {
        if !selection.strength.is_finite() {
            return Err(refuse("scale", "adapter strength must be finite"));
        }
        // The factor component: the one the selection names; else the one named like the
        // selected base component (an SDXL LoRA names its components after the base's), else
        // `adapter`, else the adapter's only component.
        let components = adapter.header().components();
        let named = |name: &str| adapter.header().component(name);
        let found = match selection.source_component.is_empty() {
            true => named(&selection.component)
                .or_else(|| named("adapter"))
                .or_else(|| (components.len() == 1).then(|| &components[0])),
            false => named(&selection.source_component),
        };
        let Some(found) = found else {
            let has: Vec<&str> = components.iter().map(|c| c.name.as_str()).collect();
            return Err(refuse(
                "components",
                format!("the adapter has no factor component for this selection; it has: {}", has.join(", ")),
            ));
        };
        let (source_component, factors) = (&found.name, &keyed(found));
        if factors.is_empty() {
            return Err(refuse("empty", "adapter contains no factors"));
        }
        let component = match selection.component.is_empty() {
            true => infer_component(header, factors)?,
            false => selection.component.clone(),
        };
        let base_tensors = &keyed(
            header.component(&component).ok_or_else(|| refuse("component", format!("base has no component {component:?}")))?,
        );
        let mut scalars = BTreeMap::new();
        for (key, tensor) in factors {
            match (validate(adapter, source_component, key, tensor)?, key.strip_suffix(".alpha")) {
                (Some(value), Some(path)) => drop(scalars.insert(path.to_string(), value)),
                (None, Some(_)) => return Err(refuse("alpha", format!("{key}: alpha must be a scalar"))),
                _ => {}
            }
        }
        graph.adapters.push(AdapterRef {
            reference: selection.manifest.id(),
            scale: selection.strength,
            kind: "lora",
            component: component.clone(),
            source_component: source_component.clone(),
            family: "",
        });
        let alias = format!("adapter_{index}");
        for (path, pair) in pairs(factors)? {
            let (Some((a_key, a)), Some((b_key, b))) = (pair.get("a"), pair.get("b")) else {
                return Err(refuse("pair", format!("{component}.{path}: incomplete A/B pair")));
            };
            let weight = base_tensors.get(&format!("{path}.weight"));
            if weight.is_some_and(|w| w.shape.len() != 2) {
                return Err(refuse("target_type", format!("{component}.{path}: only ordinary linear LoRA targets are supported")));
            }
            let fits = weight.is_some_and(|w| {
                matches!(w.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                    && a.shape.len() == 2
                    && b.shape.len() == 2
                    && a.shape[0] > 0
                    && a.shape[0] == b.shape[1]
                    && [b.shape[0], a.shape[1]] == w.shape[..]
                    && a.dtype == b.dtype
            });
            if !fits {
                return Err(refuse("shape", format!("{component}.{path}: factors do not fit the base projection")));
            }
            let rank = a.shape[0];
            let alpha = scalars.get(&path).copied().unwrap_or(rank as f64);
            if !(alpha / rank as f64 * selection.strength).is_finite() {
                return Err(refuse("scale", format!("{component}.{path}: effective scale is not finite")));
            }
            if selection.strength == 0.0 {
                continue;
            }
            for suffix in ["weight", "bias"] {
                let key = format!("{path}.{suffix}");
                if base_tensors.contains_key(&key) {
                    renamed.entry((component.clone(), key)).or_insert(format!("{path}.base_layer.{suffix}"));
                }
            }
            for (role, key) in [("A", a_key), ("B", b_key)] {
                let grafted = format!("{path}.lora_{role}.{alias}.weight");
                let take = (component.clone(), grafted, index + 1, source_component.clone(), key.clone());
                grafts.entry(component.clone()).or_default().push(take);
            }
            graph.layers.push(Linear {
                component: component.clone(),
                target: path.clone(),
                adapter: alias.clone(),
                rank,
                alpha,
                strength: selection.strength,
                dtype: a.dtype.name().into(),
            });
        }
    }
    let mut takes = vec![];
    for component in header.components() {
        let (name, mut renames) = (&component.name, vec![]);
        for key in component.tensors.iter().map(|t| &t.key) {
            match renamed.get(&(name.clone(), key.clone())) {
                Some(new) => renames.push((name.clone(), new.clone(), 0, name.clone(), key.clone())),
                None => takes.push((name.clone(), key.clone(), 0, name.clone(), key.clone())),
            }
        }
        takes.extend(renames);
        takes.extend(grafts.remove(name).unwrap_or_default());
    }
    Ok((takes, graph))
}

/// The adapter view of `base` with `selections`. A held view is reused (and its last use
/// renewed); one collected, or left unfinished by a crash, is composed again.
pub fn compose(store: &Store, base: &ObjectRef, selections: &[Selection]) -> io::Result<Composed> {
    let rows: Vec<_> = selections
        .iter()
        .map(|s| serde_json::json!([&s.manifest.id(), s.component, s.source_component, s.strength.to_string()]))
        .collect();
    let identity = tensorfs_core::sha256::hex_digest(
        &serde_json_canonicalizer::to_vec(&serde_json::json!([FORMAT, base.id(), rows])).map_err(io::Error::other)?,
    );
    let id = pin::view(&identity[..40]);
    if let Some(held) = store.get_pin(&id).map_err(storage::io_error)? {
        if let Some(manifest) = held.roots.first().filter(|m| store.reuse(m).unwrap_or(false)) {
            store.touch_pin(&id).map_err(storage::io_error)?;
            return Ok(Composed { pin: id, manifest: manifest.clone() });
        }
    }
    let checkpoint = |manifest: &ObjectRef| store.checkpoint(manifest).map_err(storage::io_error);
    let base = checkpoint(base)?;
    let adapters = selections.iter().map(|s| Ok((s.clone(), checkpoint(&s.manifest)?))).collect::<io::Result<Vec<_>>>()?;
    let (takes, graph) = plan(&base, &adapters)?;
    let mut writer = store.writer(None, Some(&id)).map_err(storage::io_error)?;
    let order: Vec<(&str, &str)> = takes.iter().map(|t| (t.0.as_str(), t.1.as_str())).collect();
    writer.set_order(&order);
    for (component, key, source, from_component, from_key) in &takes {
        let from = match source {
            0 => &base,
            i => &adapters[i - 1].1,
        };
        writer.copy_as(component, key, from, from_component, from_key).map_err(storage::io_error)?;
    }
    for (path, object) in &base.manifest().files {
        if path != tensorfs_core::format::HEADER_PATH {
            writer.file(path, &store.read(object).map_err(storage::io_error)?).map_err(storage::io_error)?;
        }
    }
    for (key, value) in &base.header().metadata {
        writer.set_metadata(key, value.clone());
    }
    let graph = serde_json_canonicalizer::to_vec(&graph).map_err(io::Error::other)?;
    writer.file(GRAPH_FILE, &graph).map_err(storage::io_error)?;
    let manifest = writer.commit().map_err(storage::io_error)?;
    Ok(Composed { pin: id, manifest })
}
