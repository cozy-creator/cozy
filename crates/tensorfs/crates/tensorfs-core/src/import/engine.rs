//! The one converter engine. A family's layout is a data table (`tensorfs-core/converters/`),
//! generated from upstream's own converter; this file recognizes a source by its tables' key
//! sets and applies a table row by row: `rename | view -> take(axis, idx) | permute -> shape`,
//! then the component's `cast`. A quantized carrier, folded into one encoded tensor (`carriers`),
//! is only renamed. Nothing model-specific lives here.

use cozy_platform::durable::FileAt;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;

use serde::Deserialize;

use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{Dtype, Encoding};

use super::source::{Group, SourceSet, Tensor};

macro_rules! table {
    ($name:literal) => {
        include_str!(concat!("../../converters/", $name, ".json"))
    };
}

/// The built-in tables.
pub const TABLES: [&str; 6] = [
    table!("sdxl.single_file"),
    table!("anima.single_file"),
    table!("h3.native"),
    table!("h3.lora"),
    table!("sdxl.lora"),
    table!("seedvr2"),
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    pub name: String,
    /// A pinned repository (`hf://org/repo@<commit>`) whose index, configs and tokenizers the
    /// output takes when its source carries none.
    #[serde(default)]
    pub reference: Option<String>,
    /// Never chosen by recognition alone: the caller names it (layouts whose headers match
    /// another layout's byte-for-byte).
    #[serde(default)]
    pub explicit: bool,
    /// A LoRA table: rows are modules, any non-empty subset may be present.
    #[serde(default)]
    pub lora: Option<Lora>,
    /// Documents the output carries, by path.
    #[serde(default)]
    pub files: BTreeMap<String, serde_json::Value>,
    pub components: Vec<Component>,
}

/// Source key suffixes of each factor; the output spells them PEFT's way.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lora {
    pub a: Vec<String>,
    pub b: Vec<String>,
    pub alpha: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub name: String,
    /// The source keys this component owns in a shared file (longest prefix wins).
    #[serde(default)]
    pub prefix: String,
    /// Output names by source folder suffix, when one layout lands under several names.
    #[serde(default)]
    pub folders: BTreeMap<String, String>,
    #[serde(default)]
    pub cast: Option<Dtype>,
    /// Source keys every matching source has and the output does not take.
    #[serde(default)]
    pub drop: Vec<String>,
    /// Source keys a source may have; never taken.
    #[serde(default)]
    pub optional: Vec<String>,
    /// Shape preconditions on source keys.
    #[serde(default)]
    pub expect: BTreeMap<String, Vec<u64>>,
    pub rows: Vec<Row>,
}

#[derive(Debug, Deserialize)]
#[serde(from = "RowRepr")]
pub struct Row {
    pub target: String,
    pub source: Src,
    pub ops: Ops,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RowRepr {
    Same((String,)),
    Rename((String, Src)),
    Op((String, Src, Ops)),
}

impl From<RowRepr> for Row {
    fn from(r: RowRepr) -> Row {
        let (target, source, ops) = match r {
            RowRepr::Same((k,)) => (k.clone(), Src::Key(k), Ops::default()),
            RowRepr::Rename((t, s)) => (t, s, Ops::default()),
            RowRepr::Op((t, s, o)) => (t, s, o),
        };
        Row {
            target,
            source,
            ops,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Src {
    Key(String),
    /// A byte range of a file the source set holds, found by its digest (no parser runs).
    Raw(Raw),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Raw {
    pub sha256: String,
    pub length: u64,
    pub offset: u64,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ops {
    #[serde(default)]
    pub view: Option<Vec<i64>>,
    #[serde(default)]
    pub take: Option<(usize, Vec<u64>)>,
    #[serde(default)]
    pub permute: Option<Vec<usize>>,
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
    /// LoRA rows: the base module's `[in, out]`, so A is `[r, in]` and B `[out, r]`.
    #[serde(default)]
    pub io: Option<(u64, u64)>,
    /// LoRA rows: the ops applied to B.
    #[serde(default)]
    pub b: Option<Box<Ops>>,
}

/// Every built-in table, parsed once.
pub fn tables() -> &'static [Table] {
    static PARSED: std::sync::OnceLock<Vec<Table>> = std::sync::OnceLock::new();
    PARSED.get_or_init(|| {
        TABLES
            .iter()
            .map(|text| serde_json::from_str(text).expect("a built-in converter table"))
            .collect()
    })
}

// ------------------------------------------------------------------------------ recognition

/// One table component matched to one source group.
#[derive(Debug, Clone)]
pub struct Claim {
    pub table: usize,
    pub component: usize,
    pub group: usize,
    /// The output component's name.
    pub name: String,
}

#[derive(Debug, Default)]
pub struct Recognized {
    pub claims: Vec<Claim>,
    /// Table indices used, in order.
    pub tables: Vec<usize>,
    /// Groups no table claims: stored as they are.
    pub unclaimed: Vec<usize>,
}

fn lora_split<'k>(lora: &Lora, key: &'k str) -> Option<(&'k str, Factor)> {
    let mut best: Option<(&str, Factor)> = None;
    for (suffixes, factor) in [
        (&lora.a, Factor::A),
        (&lora.b, Factor::B),
        (&lora.alpha, Factor::Alpha),
    ] {
        for s in suffixes {
            if let Some(stem) = key.strip_suffix(s.as_str()) {
                if best.is_none_or(|(b, _)| stem.len() < b.len()) {
                    best = Some((stem, factor));
                }
            }
        }
    }
    best
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Factor {
    A,
    B,
    Alpha,
}

/// Why one bucket of keys is not one component's layout: unknown and missing keys.
fn mismatch(
    table: &Table,
    c: &Component,
    keys: &BTreeSet<&str>,
) -> Option<(Vec<String>, Vec<String>)> {
    if let Some(lora) = &table.lora {
        let stems: BTreeSet<&str> = c.rows.iter().filter_map(|r| r.key()).collect();
        let unknown: Vec<String> = keys
            .iter()
            .filter(|k| !lora_split(lora, k).is_some_and(|(stem, _)| stems.contains(stem)))
            .map(|k| k.to_string())
            .collect();
        return (!unknown.is_empty() || keys.is_empty()).then_some((unknown, vec![]));
    }
    let required: BTreeSet<&str> = c
        .rows
        .iter()
        .filter_map(|r| r.key())
        .chain(c.drop.iter().map(String::as_str))
        .collect();
    let unknown: Vec<String> = keys
        .iter()
        .filter(|k| !required.contains(**k) && !c.optional.iter().any(|o| o == **k))
        .map(|k| k.to_string())
        .collect();
    let missing: Vec<String> = required
        .iter()
        .filter(|k| !keys.contains(**k))
        .map(|k| k.to_string())
        .collect();
    (!unknown.is_empty() || !missing.is_empty()).then_some((unknown, missing))
}

impl Row {
    fn key(&self) -> Option<&str> {
        match &self.source {
            Src::Key(k) => Some(k),
            Src::Raw(_) => None,
        }
    }
}

/// How `table` projects `group`: each prefix bucket of its keys, matched to the one
/// component whose layout it is. `Err` names the closest mismatch.
fn project(table: &Table, group: &Group) -> std::result::Result<Vec<usize>, String> {
    let prefixes: BTreeSet<&str> = table
        .components
        .iter()
        .filter(|c| c.rows.iter().any(|r| r.key().is_some()))
        .map(|c| c.prefix.as_str())
        .collect();
    let mut buckets: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for key in group.tensors.keys() {
        let Some(prefix) = prefixes
            .iter()
            .filter(|p| key.starts_with(**p))
            .max_by_key(|p| p.len())
        else {
            return Err(format!("{key:?} is under no component prefix"));
        };
        buckets
            .entry(prefix)
            .or_default()
            .insert(&key[prefix.len()..]);
    }
    let mut out = Vec::new();
    for (prefix, keys) in &buckets {
        let mut closest = (usize::MAX, String::new());
        let fits: Vec<usize> = (0..table.components.len())
            .filter(|i| {
                let c = &table.components[*i];
                if c.prefix != *prefix {
                    return false;
                }
                match mismatch(table, c, keys) {
                    None => true,
                    Some((unknown, missing)) => {
                        if unknown.len() + missing.len() < closest.0 {
                            let detail = format!(
                                "{}: {} unknown {:?}, {} missing {:?}",
                                c.name,
                                unknown.len(),
                                &unknown[..unknown.len().min(3)],
                                missing.len(),
                                &missing[..missing.len().min(3)]
                            );
                            closest = (unknown.len() + missing.len(), detail);
                        }
                        false
                    }
                }
            })
            .collect();
        match fits.as_slice() {
            [one] => out.push(*one),
            [] => return Err(closest.1),
            _ => {
                return Err(format!(
                    "keys under {prefix:?} fit {} components",
                    fits.len()
                ))
            }
        }
    }
    Ok(out)
}

fn output_name(c: &Component, folder: &str) -> String {
    c.folders
        .iter()
        .find(|(suffix, _)| folder == suffix.as_str() || folder.ends_with(&format!("/{suffix}")))
        .map(|(_, name)| name.clone())
        .unwrap_or_else(|| c.name.clone())
}

/// Which tables claim which groups. `named` restricts recognition to that table and makes
/// a mismatch a refusal; otherwise an unmatched group is simply unclaimed.
pub fn recognize(set: &SourceSet, tables: &[Table], named: Option<&str>) -> Result<Recognized> {
    let candidates: Vec<usize> = match named {
        Some(name) => match tables.iter().position(|t| t.name == name) {
            Some(i) => vec![i],
            None => {
                let names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("no converter {name:?}; built in: {names:?}"),
                );
            }
        },
        None => (0..tables.len()).collect(),
    };
    let mut out = Recognized::default();
    let mut owner: BTreeMap<usize, usize> = BTreeMap::new();
    for t in candidates {
        let table = &tables[t];
        let mut claims = Vec::new();
        let mut why = Vec::new();
        for (g, group) in set.groups.iter().enumerate() {
            match project(table, group) {
                Ok(components) => claims.extend(components.into_iter().map(|c| Claim {
                    table: t,
                    component: c,
                    group: g,
                    name: output_name(&table.components[c], &group.folder),
                })),
                Err(e) => why.push(format!("{}: {e}", group.label())),
            }
        }
        let keyed = |c: &Component| c.rows.iter().any(|r| r.key().is_some());
        let complete = match table.lora {
            Some(_) => !claims.is_empty(),
            None => table
                .components
                .iter()
                .enumerate()
                .filter(|(_, c)| keyed(c))
                .all(|(i, _)| claims.iter().any(|claim| claim.component == i)),
        };
        let raw = table
            .components
            .iter()
            .all(|c| keyed(c) || raw_present(set, c));
        if !(complete && raw) {
            if named.is_some() {
                return refuse(
                    Code::MISSING_TENSOR,
                    format!(
                        "the source is not the {} layout: {}",
                        table.name,
                        why.join("; ")
                    ),
                );
            }
            continue;
        }
        if table.explicit && named.is_none() {
            return refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!(
                    "the source matches {0}, whose headers are identical to other layouts'; \
                     name it to convert ({0}), or import it as-is",
                    table.name
                ),
            );
        }
        let mut names = BTreeSet::new();
        for claim in &claims {
            if let Some(other) = owner.insert(claim.group, t).filter(|o| *o != t) {
                return refuse(
                    Code::AMBIGUOUS_CLASSIFICATION,
                    format!(
                        "{} matches both {} and {}",
                        set.groups[claim.group].label(),
                        tables[other].name,
                        table.name
                    ),
                );
            }
            if !names.insert(claim.name.clone()) && table.lora.is_none() {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("{}: two sources for component {}", table.name, claim.name),
                );
            }
        }
        out.claims.extend(claims);
        out.tables.push(t);
    }
    out.unclaimed = (0..set.groups.len())
        .filter(|g| !owner.contains_key(g))
        .collect();
    Ok(out)
}

fn raw_present(set: &SourceSet, c: &Component) -> bool {
    c.rows.iter().all(|r| match &r.source {
        Src::Raw(raw) => set.find_raw(&raw.sha256, raw.length).is_some(),
        Src::Key(_) => true,
    })
}

// ------------------------------------------------------------------------------ planning

/// One output tensor and how its bytes are read.
#[derive(Debug, Clone)]
pub struct Planned {
    pub component: String,
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub encoding: Option<Encoding>,
    /// A plain tensor's one part (role `""`), or an encoded tensor's parts in order.
    pub parts: Vec<PlannedPart>,
}

#[derive(Debug, Clone)]
pub struct PlannedPart {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub read: Reading,
}

#[derive(Debug, Clone)]
pub struct Reading {
    pub file: usize,
    /// The source tensor's first byte in `file`, and its length.
    pub start: u64,
    pub len: u64,
    /// Bytes per run unit: the element, or one byte for a tensor read as it is.
    pub unit: u64,
    pub runs: Runs,
    pub source: Dtype,
    pub cast: Option<Dtype>,
}

impl Reading {
    /// A source range read as it is.
    fn whole(t: &Tensor) -> Reading {
        Reading {
            file: t.file,
            start: t.start,
            len: t.len,
            unit: 1,
            runs: Runs {
                run: t.len,
                base: 0,
                outer: vec![],
            },
            source: t.dtype.clone(),
            cast: None,
        }
    }
}

/// The output as contiguous source runs: `base + Σ index·stride` over the outer dims.
#[derive(Debug, Clone)]
pub struct Runs {
    pub run: u64,
    base: u64,
    outer: Vec<Dim>,
}

#[derive(Debug, Clone)]
struct Dim {
    size: u64,
    stride: u64,
    idx: Option<Vec<u64>>,
}

impl Runs {
    pub fn count(&self) -> u64 {
        self.outer.iter().map(|d| d.size).product()
    }

    /// Element offset of run `n`.
    pub fn offset(&self, mut n: u64) -> u64 {
        let mut at = self.base;
        for d in self.outer.iter().rev() {
            let i = n % d.size;
            n /= d.size;
            at += d.stride * d.idx.as_ref().map_or(i, |idx| idx[i as usize]);
        }
        at
    }
}

fn resolve_dims(what: &str, dims: &[i64], numel: u64) -> Result<Vec<u64>> {
    let known: u64 = dims.iter().filter(|d| **d > 0).map(|d| *d as u64).product();
    let free = dims.iter().filter(|d| **d == -1).count();
    if dims.iter().any(|d| *d == 0 || *d < -1) || free > 1 || known == 0 {
        return refuse(Code::SHAPE_MISMATCH, format!("{what}: dims {dims:?}"));
    }
    let fill = numel / known;
    if (free == 0 && known != numel) || (free == 1 && (!numel.is_multiple_of(known) || fill == 0)) {
        return refuse(
            Code::SHAPE_MISMATCH,
            format!("{what}: {numel} elements do not view as {dims:?}"),
        );
    }
    Ok(dims
        .iter()
        .map(|d| if *d == -1 { fill } else { *d as u64 })
        .collect())
}

/// The output shape and source runs of `ops` over a contiguous `shape`.
pub fn layout(what: &str, shape: &[u64], ops: &Ops) -> Result<(Vec<u64>, Runs)> {
    let numel: u64 = shape.iter().product();
    let dims = match &ops.view {
        Some(v) => resolve_dims(what, v, numel)?,
        None => shape.to_vec(),
    };
    let mut stride = 1;
    let mut layout: Vec<Dim> = dims
        .iter()
        .rev()
        .map(|&size| {
            let d = Dim {
                size,
                stride,
                idx: None,
            };
            stride *= size;
            d
        })
        .collect();
    layout.reverse();
    let mut base = 0;
    if let Some((axis, idx)) = &ops.take {
        let Some(d) = layout.get_mut(*axis) else {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{what}: take axis {axis} of {dims:?}"),
            );
        };
        if idx.is_empty() || idx.iter().any(|i| *i >= d.size) {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{what}: take {idx:?} of {}", d.size),
            );
        }
        if idx.windows(2).all(|w| w[1] == w[0] + 1) {
            base += idx[0] * d.stride;
        } else {
            d.idx = Some(idx.clone());
        }
        d.size = idx.len() as u64;
    }
    if let Some(p) = &ops.permute {
        let mut seen = p.clone();
        seen.sort();
        if seen != (0..layout.len()).collect::<Vec<_>>() {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{what}: permute {p:?} of rank {}", layout.len()),
            );
        }
        layout = p.iter().map(|i| layout[*i].clone()).collect();
    }
    let taken: u64 = layout.iter().map(|d| d.size).product();
    let out = match &ops.shape {
        Some(s) => resolve_dims(what, s, taken)?,
        None => layout.iter().map(|d| d.size).collect(),
    };
    let mut run = 1;
    while let Some(d) = layout.last() {
        if d.idx.is_some() || d.stride != run {
            break;
        }
        run *= d.size;
        layout.pop();
    }
    Ok((
        out,
        Runs {
            run,
            base,
            outer: layout,
        },
    ))
}

/// The output dtype, and the narrowing that reaches it if any.
fn cast_to(what: &str, from: &Dtype, to: Option<&Dtype>) -> Result<(Dtype, Option<Dtype>)> {
    match to {
        None => Ok((from.clone(), None)),
        Some(to) if to == from => Ok((from.clone(), None)),
        Some(Dtype::F16) if matches!(from, Dtype::F32 | Dtype::BF16) => {
            Ok((Dtype::F16, Some(Dtype::F16)))
        }
        Some(to) => refuse(
            Code::DTYPE_MISMATCH,
            format!("{what}: no narrowing from {from} to {to}"),
        ),
    }
}

/// `component/key` read from `t` through `ops` and `cast`. A tensor whose bytes do not
/// move is one run of bytes, so it may be any dtype. An encoded tensor is only renamed: its
/// parts are the producer's bytes, and `cast` names its logical dtype.
fn planned(
    component: &str,
    key: &str,
    t: &Tensor,
    ops: &Ops,
    cast: Option<&Dtype>,
) -> Result<Planned> {
    let what = format!("{component}/{key}");
    if let Some(folded) = &t.encoded {
        if ops.view.is_some() || ops.take.is_some() || ops.permute.is_some() || ops.shape.is_some()
        {
            return refuse(
                Code::SEAM_UNBANKED,
                format!(
                    "{what}: a {} tensor; a table row may rename it, never cut or reshape it",
                    folded.encoding.scheme
                ),
            );
        }
        let parts = (folded.parts.iter())
            .map(|p| PlannedPart {
                role: p.role.clone(),
                dtype: p.dtype.clone(),
                shape: p.shape.clone(),
                read: Reading::whole(&p.at),
            })
            .collect();
        return Ok(Planned {
            component: component.into(),
            key: key.into(),
            dtype: cast.unwrap_or(&t.dtype).clone(),
            shape: t.shape.clone(),
            encoding: Some(folded.encoding.clone()),
            parts,
        });
    }
    let (dtype, cast) = cast_to(&what, &t.dtype, cast)?;
    let (shape, runs) = layout(&what, &t.shape, ops)?;
    let moves = ops.view.is_some() || ops.take.is_some() || ops.permute.is_some();
    let read = match (moves || cast.is_some(), t.dtype.bits()) {
        (true, Some(bits)) if bits.is_multiple_of(8) => Reading {
            file: t.file,
            start: t.start,
            len: t.len,
            unit: bits / 8,
            runs,
            source: t.dtype.clone(),
            cast,
        },
        (true, _) => {
            return refuse(
                Code::DTYPE_MISMATCH,
                format!("{what}: {} elements are not whole bytes", t.dtype),
            )
        }
        (false, _) => Reading::whole(t),
    };
    Ok(Planned {
        component: component.into(),
        key: key.into(),
        dtype: dtype.clone(),
        shape: shape.clone(),
        encoding: None,
        parts: vec![PlannedPart {
            role: String::new(),
            dtype,
            shape,
            read,
        }],
    })
}

/// Every output tensor, in construction order: claimed groups through their tables' rows in
/// row order (the target's `state_dict()` order), then unclaimed groups as they are, under
/// `identity(group)`'s component name.
pub fn plan(
    set: &SourceSet,
    tables: &[Table],
    r: &Recognized,
    identity: &dyn Fn(&Group) -> String,
) -> Result<Vec<Planned>> {
    let mut out = Vec::new();
    for &t in &r.tables {
        let table = &tables[t];
        for (ci, c) in table.components.iter().enumerate() {
            let claims: Vec<&Claim> = r
                .claims
                .iter()
                .filter(|x| x.table == t && x.component == ci)
                .collect();
            if claims.is_empty() && c.rows.iter().all(|row| row.key().is_none()) {
                plan_raw(set, c, &mut out)?;
            }
            for claim in claims {
                let group = &set.groups[claim.group];
                let local = |k: &str| group.tensors.get(&format!("{}{k}", c.prefix));
                for (key, shape) in &c.expect {
                    if let Some(t) = local(key).filter(|t| &t.shape != shape) {
                        return refuse(
                            Code::SHAPE_MISMATCH,
                            format!(
                                "{}: {key} is {:?}, {} expects {shape:?}",
                                claim.name, t.shape, table.name
                            ),
                        );
                    }
                }
                match &table.lora {
                    Some(lora) => plan_lora(table, lora, c, claim, group, &mut out)?,
                    None => {
                        for row in &c.rows {
                            let Src::Key(k) = &row.source else { continue };
                            let t = local(k).expect("recognition checked the key set");
                            out.push(planned(
                                &claim.name,
                                &row.target,
                                t,
                                &row.ops,
                                c.cast.as_ref(),
                            )?);
                        }
                    }
                }
            }
        }
    }
    // Unclaimed groups are stored as they are, in the construction order of the class their
    // config names; keys its reference lacks keep the order of their bytes in the source.
    for &g in &r.unclaimed {
        let group = &set.groups[g];
        let name = identity(group);
        let keys: Vec<(&str, (usize, u64))> = (group.tensors.iter())
            .map(|(k, t)| (k.as_str(), (t.file, t.start)))
            .collect();
        let class = set.class_of(group);
        let ordered = super::constructor_order(class.as_deref(), &keys);
        if !ordered.unknown.is_empty() {
            tracing::warn!(
                "{name}: {} of {} tensors have no constructor order for {}; they keep the source's order",
                ordered.unknown.len(),
                keys.len(),
                class.as_deref().unwrap_or("an unnamed class"),
            );
        }
        for i in ordered.order {
            let key = keys[i].0;
            out.push(planned(
                &name,
                key,
                &group.tensors[key],
                &Ops::default(),
                None,
            )?);
        }
    }
    let mut seen = BTreeSet::new();
    for p in &out {
        if !seen.insert((p.component.as_str(), p.key.as_str())) {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("{}/{} is produced twice", p.component, p.key),
            );
        }
    }
    Ok(out)
}

fn plan_raw(set: &SourceSet, c: &Component, out: &mut Vec<Planned>) -> Result<()> {
    for row in &c.rows {
        let Src::Raw(raw) = &row.source else { continue };
        let (file, len) = set
            .find_raw(&raw.sha256, raw.length)
            .expect("recognition found it");
        let bytes = raw.dtype.byte_len(&raw.shape).unwrap_or(u64::MAX);
        if raw.offset.checked_add(bytes).is_none_or(|end| end > len) {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{}/{}: range outside its file", c.name, row.target),
            );
        }
        let t = Tensor {
            dtype: raw.dtype.clone(),
            shape: raw.shape.clone(),
            file,
            start: raw.offset,
            len: bytes,
            encoded: None,
        };
        out.push(planned(
            &c.name,
            &row.target,
            &t,
            &row.ops,
            c.cast.as_ref(),
        )?);
    }
    Ok(())
}

fn plan_lora(
    table: &Table,
    lora: &Lora,
    c: &Component,
    claim: &Claim,
    group: &Group,
    out: &mut Vec<Planned>,
) -> Result<()> {
    let mut factors: BTreeMap<&str, [Option<(&str, &Tensor)>; 3]> = BTreeMap::new();
    let longest = |key: &str| {
        table
            .components
            .iter()
            .map(|o| o.prefix.as_str())
            .filter(|p| key.starts_with(p))
            .max_by_key(|p| p.len())
    };
    for (key, t) in group
        .tensors
        .iter()
        .filter(|(k, _)| longest(k) == Some(c.prefix.as_str()))
    {
        let local = &key[c.prefix.len()..];
        let (stem, factor) = lora_split(lora, local).expect("recognition checked the key set");
        let slot = &mut factors.entry(stem).or_default()[factor as usize];
        if slot.replace((key, t)).is_some() {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("{key}: a second {factor:?} factor for {stem}"),
            );
        }
    }
    for row in &c.rows {
        let Some([a, b, alpha]) = row.key().and_then(|k| factors.get(k)) else {
            continue;
        };
        let (Some((ka, a)), Some((_, b))) = (a, b) else {
            return refuse(
                Code::MISSING_COMPANION_ROLE,
                format!("{}: LoRA A and B are both required", row.target),
            );
        };
        let Some((input, output)) = row.ops.io else {
            return refuse(
                Code::MISSING_FIELD,
                format!("{}: LoRA row without io", table.name),
            );
        };
        let float = |d: &Dtype| matches!(d, Dtype::F16 | Dtype::BF16 | Dtype::F32);
        let rank = a.shape.first().copied().unwrap_or(0);
        if a.encoded.is_some()
            || b.encoded.is_some()
            || a.shape.len() != 2
            || rank == 0
            || rank > input.min(output)
            || a.shape[1] != input
            || b.shape != [output, rank]
            || a.dtype != b.dtype
            || !float(&a.dtype)
            || alpha.is_some_and(|(_, t)| !t.shape.is_empty() || !float(&t.dtype))
        {
            return refuse(
                Code::SHAPE_MISMATCH,
                format!("{ka}: LoRA factors must be float A [r,{input}], B [{output},r] and a scalar alpha"),
            );
        }
        let none = Ops::default();
        for (suffix, t, ops) in [
            (".lora_A.weight", Some(*a), &none),
            (
                ".lora_B.weight",
                Some(*b),
                row.ops.b.as_deref().unwrap_or(&none),
            ),
            (".alpha", alpha.map(|(_, t)| t), &none),
        ] {
            let Some(t) = t else { continue };
            let key = format!("{}{suffix}", row.target);
            out.push(planned(&claim.name, &key, t, ops, None)?);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------ bytes

/// Streams one planned part's bytes out of its source file.
pub struct TensorReader<'f> {
    file: &'f File,
    r: &'f Reading,
    what: String,
    /// Next run, and units of it already read.
    run: u64,
    within: u64,
    /// The source tensor whole, when its runs are too short to read one by one.
    whole: Option<Vec<u8>>,
    raw: Vec<u8>,
    out: Vec<u8>,
    at: usize,
    seen: u64,
}

/// Runs shorter than this are gathered from the whole source tensor in memory.
const SHORT_RUN: u64 = 4096;
/// Bytes read from the file per step.
const STEP: u64 = 8 << 20;

impl<'f> TensorReader<'f> {
    /// `what` names the tensor in errors.
    pub fn new(file: &'f File, r: &'f Reading, what: String) -> Result<TensorReader<'f>> {
        let whole = if r.runs.count() > 1 && r.runs.run * r.unit < SHORT_RUN {
            let mut buf = vec![0u8; r.len as usize];
            file.read_exact_at(&mut buf, r.start)
                .map_err(|e| io(&what, e))?;
            Some(buf)
        } else {
            None
        };
        Ok(TensorReader {
            file,
            r,
            what,
            run: 0,
            within: 0,
            whole,
            raw: vec![],
            out: vec![],
            at: 0,
            seen: 0,
        })
    }

    fn fill(&mut self) -> std::io::Result<bool> {
        let r = self.r;
        if self.run >= r.runs.count() || r.runs.run == 0 {
            return Ok(false);
        }
        self.raw.clear();
        let unit = r.unit;
        if let Some(whole) = &self.whole {
            let run = (r.runs.run * unit) as usize;
            while self.run < r.runs.count() && (self.raw.len() as u64) < STEP {
                let at = (r.runs.offset(self.run) * unit) as usize;
                self.raw.extend_from_slice(&whole[at..at + run]);
                self.run += 1;
            }
        } else {
            let take = (r.runs.run - self.within).min((STEP / unit).max(1));
            let at = r.start + (r.runs.offset(self.run) + self.within) * unit;
            self.raw.resize((take * unit) as usize, 0);
            self.file.read_exact_at(&mut self.raw, at)?;
            self.within += take;
            if self.within == r.runs.run {
                self.run += 1;
                self.within = 0;
            }
        }
        self.at = 0;
        if r.cast.is_none() {
            std::mem::swap(&mut self.out, &mut self.raw);
            return Ok(true);
        }
        self.out.clear();
        let wide = r.source == Dtype::F32;
        for e in self.raw.chunks_exact(unit as usize) {
            let bits = if wide {
                u32::from_le_bytes([e[0], e[1], e[2], e[3]])
            } else {
                u32::from(u16::from_le_bytes([e[0], e[1]])) << 16
            };
            let half = f32_to_f16(bits);
            if half & 0x7fff == 0x7c00 && bits & 0x7f80_0000 != 0x7f80_0000 {
                return Err(std::io::Error::other(format!(
                    "{}: element {} ({}) overflows F16",
                    self.what,
                    self.seen,
                    f32::from_bits(bits)
                )));
            }
            self.seen += 1;
            self.out.extend_from_slice(&half.to_le_bytes());
        }
        Ok(true)
    }
}

impl Read for TensorReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while self.at == self.out.len() {
            if !self.fill()? {
                return Ok(0);
            }
        }
        let n = buf.len().min(self.out.len() - self.at);
        buf[..n].copy_from_slice(&self.out[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

fn io(what: &str, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{what}: {e}"),
    }
}

/// numpy's `astype(float16)` from f32 bits: round to nearest even, NaN payload kept.
pub fn f32_to_f16(f: u32) -> u16 {
    let sign = ((f & 0x8000_0000) >> 16) as u16;
    let exp = f & 0x7f80_0000;
    if exp >= 0x4780_0000 {
        let sig = f & 0x007f_ffff;
        if exp == 0x7f80_0000 && sig != 0 {
            let nan = 0x7c00 + (sig >> 13) as u16;
            return sign + if nan == 0x7c00 { nan + 1 } else { nan };
        }
        return sign + 0x7c00;
    }
    if exp <= 0x3800_0000 {
        if exp < 0x3300_0000 {
            return sign;
        }
        let mut sig = (0x0080_0000 + (f & 0x007f_ffff)) >> (113 - (exp >> 23));
        if (sig & 0x3fff) != 0x1000 || (f & 0x7ff) != 0 {
            sig += 0x1000;
        }
        return sign + (sig >> 13) as u16;
    }
    let mut sig = f & 0x007f_ffff;
    if (sig & 0x3fff) != 0x1000 {
        sig += 0x1000;
    }
    sign + ((sig >> 13) as u16 + ((exp - 0x3800_0000) >> 13) as u16)
}
