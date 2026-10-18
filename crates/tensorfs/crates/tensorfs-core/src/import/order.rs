//! Construction order for a source already in the canonical layout (spec §10). Safetensors files
//! sort their keys, so the order comes from the model class's generated reference
//! (`orders/<class>.json`, `scripts/orders.py`): for each parent pattern, its children in the
//! order the class's `state_dict()` gives them. Nothing here sorts names.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

macro_rules! orders {
    ($($class:literal),* $(,)?) => {
        [$(($class, include_str!(concat!("../../orders/", $class, ".json")))),*]
    };
}

/// Every class with a reference, by its `_class_name` (Diffusers) or `architectures[0]`
/// (Transformers).
const ORDERS: [(&str, &str); 17] = orders![
    "AnimaTextConditioner",
    "AutoencoderKL",
    "AutoencoderKLFlux2",
    "AutoencoderKLQwenImage",
    "AutoencoderKLQwenImage21",
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "CosmosTransformer3DModel",
    "Flux3Transformer2DModel",
    "MiniMaxH3Transformer3DModel",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3Model",
    "Qwen3VLForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "QwenImage21Transformer2DModel",
    "QwenImageTransformer2DModel",
    "UNet2DConditionModel",
];

#[derive(Deserialize)]
struct Reference {
    /// Parent pattern (digit segments `#`, `""` for the root) → its children in order.
    children: HashMap<String, Vec<String>>,
}

fn reference(class: &str) -> Option<&'static Reference> {
    static REFS: OnceLock<HashMap<&'static str, Reference>> = OnceLock::new();
    let refs = REFS.get_or_init(|| {
        (ORDERS.iter())
            .map(|(class, json)| {
                let parsed = serde_json::from_str(json).expect("orders are generated");
                (*class, parsed)
            })
            .collect()
    });
    refs.get(class)
}

/// The classes with a reference.
pub fn classes() -> impl Iterator<Item = &'static str> {
    ORDERS.iter().map(|(class, _)| *class)
}

/// `keys` in a class's construction order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ordered {
    /// Positions into the keys, in construction order.
    pub order: Vec<usize>,
    /// Positions of keys whose path the reference does not hold (or every key, without a
    /// reference): they keep the source's data order after the known children of their parent.
    pub unknown: Vec<usize>,
}

/// `keys` in `class`'s construction order. `data` is each key's place in the source (the order
/// of its bytes). Indices ascend; a child the reference lacks, or every key of a class without
/// a reference, keeps the data order and is reported in `unknown`.
pub fn constructor_order<D: Ord + Copy>(class: Option<&str>, keys: &[(&str, D)]) -> Ordered {
    let reference = class.and_then(reference);
    let paths: Vec<Vec<&str>> = keys.iter().map(|(k, _)| k.split('.').collect()).collect();
    let mut order = Vec::with_capacity(keys.len());
    arrange(
        keys,
        &paths,
        (0..keys.len()).collect(),
        0,
        "",
        reference,
        &mut order,
    );
    let unknown = (0..keys.len())
        .filter(|i| reference.is_none_or(|r| !r.covers(&paths[*i])))
        .collect();
    Ordered { order, unknown }
}

fn index(segment: &str) -> bool {
    !segment.is_empty() && segment.bytes().all(|b| b.is_ascii_digit())
}

fn child(segment: &str) -> &str {
    if index(segment) {
        "#"
    } else {
        segment
    }
}

fn join(parent: &str, segment: &str) -> String {
    match parent {
        "" => child(segment).to_owned(),
        p => format!("{p}.{}", child(segment)),
    }
}

impl Reference {
    fn covers(&self, path: &[&str]) -> bool {
        let mut parent = String::new();
        for segment in path {
            if self.rank(&parent, segment).is_none() {
                return false;
            }
            parent = join(&parent, segment);
        }
        true
    }

    fn rank(&self, parent: &str, segment: &str) -> Option<usize> {
        let children = self.children.get(parent)?;
        children.iter().position(|c| c == child(segment))
    }
}

/// Emits `members` (keys sharing their first `depth` segments) in order: a key ending here,
/// then each child's subtree by the reference's rank, indices by value, unknown children by
/// their first byte in the source.
fn arrange<D: Ord + Copy>(
    keys: &[(&str, D)],
    paths: &[Vec<&str>],
    members: Vec<usize>,
    depth: usize,
    parent: &str,
    reference: Option<&Reference>,
    out: &mut Vec<usize>,
) {
    let mut leaves: Vec<usize> = Vec::new();
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    for i in members {
        match paths[i].get(depth) {
            None => leaves.push(i),
            Some(segment) => match groups.iter_mut().find(|(s, _)| s == segment) {
                Some((_, list)) => list.push(i),
                None => groups.push((segment, vec![i])),
            },
        }
    }
    leaves.sort_by_key(|i| keys[*i].1);
    out.extend(leaves);
    let first = |list: &[usize]| list.iter().map(|i| keys[*i].1).min();
    groups.sort_by_cached_key(|(segment, list)| {
        let rank = reference.and_then(|r| r.rank(parent, segment));
        let value = match index(segment) {
            true => {
                let digits = segment.trim_start_matches('0');
                (digits.len(), digits.to_owned())
            }
            false => (0, String::new()),
        };
        (rank.is_none(), rank, value, first(list))
    });
    for (segment, list) in groups {
        let next = join(parent, segment);
        arrange(keys, paths, list, depth + 1, &next, reference, out);
    }
}
