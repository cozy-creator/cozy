//! Caller LoRA adapters as one view over a base checkpoint (the Python worker's
//! `lora_composition.prepare`): base `<path>.weight/.bias` become `<path>.base_layer.*`, factors
//! graft in as `<path>.lora_{A,B}.adapter_<i>.weight`, and each adapted component's asset
//! `model_adapters.json` holds the ordered graph the executor binds with PEFT. Nothing is computed, and large parts keep their
//! objects. Factors must already be canonical PEFT keys (`<path>.lora_A.weight`,
//! `<path>.lora_B.weight`, optional scalar `<path>.alpha`): foreign layouts are normalized when a
//! model is imported, never here. A view is pinned `view-<id>`, a cache that expires like the
//! machine's others (`storage::expire`).
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
};

use serde::Serialize;
use tensorfs_core::format::{Body, Component, Dtype, Header, ObjectRef, Tensor};
use tensorfs_core::store2::{Checkpoint, Store};

use crate::storage::{self, pin};

pub const FORMAT: &str = "cozy.model.lora/1";
/// The graph's asset in each adapted component of the view.
pub const GRAPH_FILE: &str = "model_adapters.json";

/// One caller adapter, in order. An empty `component` is inferred when exactly one base
/// component fits every factor pair; an empty `source_component` is the adapter's component
/// named like `component`, else `adapter`, else its only one.
#[derive(Clone, Debug)]
pub struct Selection {
    pub checkpoint: ObjectRef,
    pub component: String,
    pub source_component: String,
    pub strength: f64,
}

/// The composed checkpoint and the pin that keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Composed {
    pub pin: String,
    pub checkpoint: ObjectRef,
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
    if base.components().iter().any(|c| c.assets.contains_key(GRAPH_FILE)) {
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
            reference: selection.checkpoint.id(),
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
        .map(|s| serde_json::json!([&s.checkpoint.id(), s.component, s.source_component, s.strength.to_string()]))
        .collect();
    let identity = tensorfs_core::sha256::hex_digest(
        &serde_json_canonicalizer::to_vec(&serde_json::json!([FORMAT, base.id(), rows])).map_err(io::Error::other)?,
    );
    let id = pin::view(&identity[..40]);
    if let Some(held) = store.get_pin(&id).map_err(storage::io_error)? {
        if let Some(header) = held.roots.first().filter(|h| store.reuse(h).unwrap_or(false)) {
            store.touch_pin(&id).map_err(storage::io_error)?;
            return Ok(Composed { pin: id, checkpoint: header.clone() });
        }
    }
    let checkpoint = |header: &ObjectRef| store.checkpoint(header).map_err(storage::io_error);
    let base = checkpoint(base)?;
    let adapters = selections.iter().map(|s| Ok((s.clone(), checkpoint(&s.checkpoint)?))).collect::<io::Result<Vec<_>>>()?;
    let (takes, graph) = plan(&base, &adapters)?;
    let adapted: BTreeSet<&str> = graph.adapters.iter().map(|a| a.component.as_str()).collect();
    let graph = serde_json_canonicalizer::to_vec(&graph).map_err(io::Error::other)?;
    let mut writer = store.writer(None, Some(&id)).map_err(storage::io_error)?;
    if let Some(pipeline) = base.pipeline() {
        writer.pipeline(&pipeline.class, pipeline.config.clone());
    }
    for component in base.components() {
        let mut declared = (writer.component(&component.name, &component.class, component.config.clone(), component.config_file.as_deref()))
            .map_err(storage::io_error)?;
        for (name, object) in &component.assets {
            declared.reuse_asset(name, object).map_err(storage::io_error)?;
        }
        if adapted.contains(component.name.as_str()) {
            declared.asset(GRAPH_FILE, &graph).map_err(storage::io_error)?;
        }
    }
    let order: Vec<(&str, &str)> = takes.iter().map(|t| (t.0.as_str(), t.1.as_str())).collect();
    writer.set_order(&order);
    for (component, key, source, from_component, from_key) in &takes {
        let from = match source {
            0 => &base,
            i => &adapters[i - 1].1,
        };
        writer.get(component).copy_as(key, from, from_component, from_key).map_err(storage::io_error)?;
    }
    for (key, value) in &base.header().metadata {
        writer.set_metadata(key, value.clone());
    }
    let header = writer.commit().map_err(storage::io_error)?;
    Ok(Composed { pin: id, checkpoint: header })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;
    use tensorfs_core::format::{Config, ConfigValue, Value};
    use tensorfs_core::writer::{Data, Source};

    pub(crate) fn f32s(n: usize, seed: f32) -> Vec<u8> {
        (0..n).flat_map(|i| (seed + i as f32 * 0.25).to_le_bytes()).collect()
    }

    /// A real checkpoint through the writer: one component of F32 tensors, its config and a
    /// tokenizer.
    pub(crate) fn checkpoint(store: &Store, key: &str, component: &str, tensors: &[(&str, Vec<u64>, Vec<u8>)]) -> ObjectRef {
        let mut writer = store.writer(None, Some(key)).unwrap();
        let config = Config::from([("a".to_string(), ConfigValue::Int(1))]);
        let mut declared = writer.component(component, "diffusers.UNet2DConditionModel", config, None).unwrap();
        for (name, shape, bytes) in tensors {
            declared.put(name, &Dtype::F32, shape, Data::Plain(Source::Read(&mut bytes.as_slice()))).unwrap();
        }
        declared.asset("tokenizer.json", b"{}").unwrap();
        writer.set_metadata("normalized", Value::Bool(true));
        writer.commit().unwrap()
    }

    /// The chunks a tensor's bytes live in.
    fn objects(checkpoint: &Checkpoint, component: &str, key: &str) -> Vec<ObjectRef> {
        let header = checkpoint.header();
        let tensor = checkpoint.tensor(component, key).unwrap();
        let spans = tensor.locations().into_iter().flat_map(|(_, at)| header.spans(&at));
        spans.map(|span| header.chunks[span.chunk].clone()).collect()
    }

    /// An adapter view renames the base projection, grafts the factors under their adapter's
    /// name and carries the rest (config, assets, metadata); every tensor reads back
    /// bit-identical and the large base part keeps its objects. A held view is reused; a
    /// collected one composes again identically.
    #[test]
    fn an_adapter_view_grafts_factors_onto_the_base() {
        let root = std::env::temp_dir().join(format!("adapter-{}", uuid::Uuid::new_v4()));
        let store = Store::open(&root).unwrap().with_grace(Duration::ZERO);
        let (weight, bias, other) = (f32s(2048 * 2048, 0.5), f32s(2048, 1.5), f32s(8, 2.5));
        let base = checkpoint(&store, "base", "unet", &[("proj.weight", vec![2048, 2048], weight.clone()), ("proj.bias", vec![2048], bias.clone()), ("other.weight", vec![8], other.clone())]);
        let (a, b) = (f32s(2 * 2048, 0.125), f32s(2048 * 2, 0.75));
        let lora = checkpoint(&store, "lora", "adapter", &[("proj.lora_A.weight", vec![2, 2048], a.clone()), ("proj.lora_B.weight", vec![2048, 2], b.clone()), ("proj.alpha", vec![], 4.0f32.to_le_bytes().to_vec())]);
        let selection = [Selection { checkpoint: lora.clone(), component: String::new(), source_component: String::new(), strength: 1.0 }];
        let view = compose(&store, &base, &selection).unwrap();
        let composed = store.checkpoint(&view.checkpoint).unwrap();
        for (key, bytes) in [("proj.base_layer.weight", &weight), ("proj.base_layer.bias", &bias), ("other.weight", &other),
            ("proj.lora_A.adapter_0.weight", &a), ("proj.lora_B.adapter_0.weight", &b)] {
            assert_eq!(&composed.read("unet", key).unwrap(), bytes, "{key}");
        }
        assert!(composed.tensor("unet", "proj.weight").is_err());
        let held = store.checkpoint(&base).unwrap();
        assert_eq!(objects(&composed, "unet", "proj.base_layer.weight"), objects(&held, "unet", "proj.weight"));
        let unet = composed.component("unet").unwrap();
        assert_eq!((unet.class.as_str(), &unet.config), ("diffusers.UNet2DConditionModel", &held.component("unet").unwrap().config));
        assert_eq!(composed.asset("unet", "tokenizer.json").unwrap(), b"{}");
        assert_eq!(composed.header().metadata.get("normalized"), Some(&Value::Bool(true)));
        let graph: serde_json::Value = serde_json::from_slice(&composed.asset("unet", GRAPH_FILE).unwrap()).unwrap();
        assert_eq!(graph["layers"][0]["alpha"], 4.0);
        assert_eq!(graph["layers"][0]["component"], "unet");
        assert_eq!(compose(&store, &base, &selection).unwrap(), view, "a held view is reused");
        let nested = compose(&store, &view.checkpoint, &selection).unwrap_err();
        assert!(nested.to_string().starts_with("adapter_nested"), "{nested}");

        store.unpin(&view.pin).unwrap();
        store.gc().unwrap();
        assert!(!store.reuse(&view.checkpoint).unwrap(), "the view went with its pin");
        assert!(store.checkpoint(&base).unwrap().read("unet", "proj.weight").is_ok(), "the base did not");
        assert_eq!(compose(&store, &base, &selection).unwrap(), view, "composed again, identically");
        let _ = std::fs::remove_dir_all(root);
    }
}
