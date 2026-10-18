//! Construction order for a source already in the canonical layout (spec §10). Safetensors files
//! sort their keys, so the order comes from a generated reference (`converters/orders.json`,
//! `scripts/orders.py`): for each parent pattern of a model class, its children in the order the
//! class's `state_dict()` gives them. Nothing here sorts names.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Deserialize)]
struct Doc {
    references: Vec<Reference>,
}

#[derive(Deserialize)]
struct Reference {
    name: String,
    /// Parent pattern (digit segments `#`, `""` for the root) → its children in order.
    children: HashMap<String, Vec<String>>,
}

fn references() -> &'static [Reference] {
    static REFS: OnceLock<Vec<Reference>> = OnceLock::new();
    REFS.get_or_init(|| {
        let doc: Doc = serde_json::from_str(include_str!("../../converters/orders.json"))
            .expect("orders.json is generated");
        doc.references
    })
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
    /// Whether every step of `key`'s path is a child the reference knows.
    fn covers(&self, key: &str) -> bool {
        let mut parent = String::new();
        for segment in key.split('.') {
            let known = self.children.get(&parent);
            if !known.is_some_and(|c| c.iter().any(|c| c == child(segment))) {
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

/// The positions of `keys` in construction order, and the reference that gave it. `data` is
/// each key's place in the source's data. The reference is the one covering most keys, when it
/// covers more than half; children it does not know follow the known ones in data order, and
/// without a reference the whole source keeps its data order.
pub fn order<D: Ord + Copy>(keys: &[(&str, D)]) -> (Vec<usize>, Option<&'static str>) {
    let best = references()
        .iter()
        .map(|r| (keys.iter().filter(|(k, _)| r.covers(k)).count(), r))
        .max_by_key(|(n, _)| *n)
        .filter(|(n, _)| 2 * n > keys.len())
        .map(|(_, r)| r);
    let paths: Vec<Vec<&str>> = keys.iter().map(|(k, _)| k.split('.').collect()).collect();
    let mut out = Vec::with_capacity(keys.len());
    arrange(
        keys,
        &paths,
        (0..keys.len()).collect(),
        0,
        "",
        best,
        &mut out,
    );
    (out, best.map(|r| r.name.as_str()))
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
