//! `tfs dev migrate` (xs-045 U12): v1 lane checkpoints become cozytensors/1 checkpoints. It goes,
//! with [`v1`], the last v1 reader, once the live lanes have moved.
//!
//! Per checkpoint:
//! 1. The v1 manifest and header are read.
//! 2. Each v1 encoding maps to its v2 scheme. A checkpoint using one v2 dropped is listed, not
//!    converted.
//! 3. D4's canonical names come from U6's tables: H3's component folders, SDXL's `text_model.`,
//!    H3's `rope.inv_freq`.
//! 4. The writer packs the parts under 16 MiB, in v1's tensor order (the execution order), and
//!    writes v1's inline configs as files. Large parts keep their v1 grid objects and files their
//!    objects: reused by digest, never copied.
//! 5. Every tensor is checked bit-identical: a reused part by its objects, a packed part by the
//!    digest of its bytes read back from the packs against the digest of its v1 bytes. The v2
//!    order of components, tensors and parts must be v1's.
//! 6. One JSON report per checkpoint, and totals that count shared objects once.
//!
//! Without a store it is a plan: the census, the names and the new bytes, from the manifest and
//! header alone ([`hub`] fetches just those from a Hub).

pub mod hub;
pub mod v1;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

use serde_json::{json, Map, Value as Json};

use crate::failure::{fail, Kind, Result};
use crate::format::layout::{ALIGN, GRID, STANDALONE};
use crate::format::{
    Body, Dtype, Encoding, Location, ObjectRef, Release, Repo, Value, HEADER_PATH,
};
use crate::import::order::{self, constructor_order};
use crate::import::tables;
use crate::sha256::hex_digest;
use crate::store2::Store;
use crate::writer::{Data, PartData, Source, Writer};
use v1::Mapping;

/// v1's names for upstream folders (H3); the tables name a folder's canonical component.
const V1_FOLDERS: [(&str, &str); 3] = [
    ("fl2va_dit", "FL2VA/transformer"),
    ("ref2va_dit", "Ref2VA/transformer"),
    ("video_vae", "vae"),
];
/// LoRA factor suffixes, as v1 and the tables' output both spell them (PEFT).
const LORA: [&str; 3] = [".lora_A.weight", ".lora_B.weight", ".alpha"];

/// One checkpoint's report, and its object sets for totals.
pub struct Report {
    pub doc: Map<String, Json>,
    pub pass: bool,
    pub v1: BTreeSet<ObjectRef>,
    pub v2: BTreeSet<ObjectRef>,
    /// The v1 manifest, and its v2 manifest once committed.
    pub manifests: (ObjectRef, Option<ObjectRef>),
}

/// Migrates one v1 checkpoint into `store`, or plans it without one. A dry run collects what it
/// wrote once checked, so `store` is a scratch store.
pub fn lane(src: &v1::Source, store: Option<&Store>, lane: &v1::Lane, dry_run: bool) -> Report {
    let mut r = Report {
        doc: Map::new(),
        pass: false,
        v1: BTreeSet::new(),
        v2: BTreeSet::new(),
        manifests: (lane.manifest.clone(), None),
    };
    r.doc.insert("names".into(), json!(lane.names));
    match migrate(src, store, lane, dry_run, &mut r) {
        Ok(pass) => r.pass = pass,
        Err(f) => drop(r.doc.insert(
            "error".into(),
            json!({"code": f.kind.code(), "detail": f.detail}),
        )),
    }
    r.doc.insert("pass".into(), r.pass.into());
    r
}

/// Totals over checkpoints; an object several share counts once.
pub fn totals(reports: &[Report]) -> Json {
    let names = |keep: &dyn Fn(&Report) -> bool| -> Vec<Json> {
        (reports.iter().filter(|r| keep(r)))
            .map(|r| r.doc["names"].clone())
            .collect()
    };
    let union = |set: &dyn Fn(&Report) -> &BTreeSet<ObjectRef>| -> BTreeSet<&ObjectRef> {
        reports.iter().flat_map(|r| set(r).iter()).collect()
    };
    let (v1, v2) = (union(&|r| &r.v1), union(&|r| &r.v2));
    let new: Vec<&&ObjectRef> = v2.iter().filter(|o| !v1.contains(**o)).collect();
    let sum =
        |objects: &mut dyn Iterator<Item = &ObjectRef>| objects.map(|o| o.length).sum::<u64>();
    let per_checkpoint: u64 = reports
        .iter()
        .filter_map(|r| r.doc.get("new")?["bytes"].as_u64())
        .sum();
    let order: BTreeMap<&str, Vec<Json>> = ["sorted", "mixed", "unknown"]
        .into_iter()
        .map(|class| {
            (
                class,
                names(&|r| r.doc.get("order").is_some_and(|o| o["class"] == class)),
            )
        })
        .collect();
    json!({
        "checkpoints": reports.len(),
        "passed": reports.iter().filter(|r| r.pass).count(),
        "failed": names(&|r| !r.pass && !r.doc.contains_key("dropped_schemes")),
        "dropped_schemes": names(&|r| r.doc.contains_key("dropped_schemes")),
        "renamed": names(&|r| r.doc.contains_key("renames")),
        "order": order,
        "v1": {"objects": v1.len(), "bytes": sum(&mut v1.iter().copied())},
        "v2": {"objects": v2.len(), "bytes": sum(&mut v2.iter().copied())},
        "new": {"objects": new.len(), "bytes": sum(&mut new.iter().map(|o| **o)), "sum_per_checkpoint": per_checkpoint},
    })
}

/// `repo`'s v2 refs file from its own v1 rows alone: each release's lanes, yanked and revision,
/// and the checkpoints no release cites, every manifest replaced by its migration. The v1
/// manifests that did not migrate when any did not.
pub fn refs(repo: &v1::Repo, reports: &[Report]) -> std::result::Result<Repo, Vec<String>> {
    let migrated: HashMap<&ObjectRef, &ObjectRef> = (reports.iter())
        .filter(|r| r.pass)
        .filter_map(|r| Some((&r.manifests.0, r.manifests.1.as_ref()?)))
        .collect();
    let mut missing = Vec::new();
    let mut v2 = |v1: &ObjectRef| match migrated.get(v1) {
        Some(m) => Some((*m).clone()),
        None => {
            missing.push(v1.sha256.clone());
            None
        }
    };
    let releases: Vec<Release> = (repo.releases.iter())
        .map(|r| Release {
            version: r.version.clone(),
            lanes: (r.lanes.iter())
                .filter_map(|(lane, m)| Some((lane.clone(), v2(m)?)))
                .collect(),
            yanked: r.yanked,
            revision: r.revision,
        })
        .collect();
    let cited: BTreeSet<&ObjectRef> = repo
        .releases
        .iter()
        .flat_map(|r| r.lanes.iter().map(|l| &l.1))
        .collect();
    let checkpoints = (repo.checkpoints.iter())
        .filter(|c| !cited.contains(c))
        .filter_map(&mut v2)
        .collect();
    match missing.is_empty() {
        true => Ok(Repo {
            name: repo.name.clone(),
            releases,
            checkpoints,
        }),
        false => Err(missing),
    }
}

fn object(o: &ObjectRef) -> Json {
    json!({"sha256": o.sha256, "length": o.length})
}

fn count(objects: &BTreeSet<ObjectRef>) -> Json {
    json!({"objects": objects.len(), "bytes": objects.iter().map(|o| o.length).sum::<u64>()})
}

fn integrity<T>(detail: impl Into<String>) -> Result<T> {
    fail(Kind::Integrity, detail)
}

fn migrate(
    src: &v1::Source,
    store: Option<&Store>,
    lane: &v1::Lane,
    dry_run: bool,
    r: &mut Report,
) -> Result<bool> {
    let m = v1::manifest(&src.read(&lane.manifest)?)?;
    let Some(header) = m.header.clone() else {
        return fail(Kind::Unsupported, "the v1 manifest has no header");
    };
    let h = v1::header(&src.read(&header)?)?;
    r.v1.extend([lane.manifest.clone(), header.clone()]);
    r.v1.extend(m.files.iter().map(|f| f.1.clone()));
    r.v1.extend(h.assets.iter().flat_map(|a| a.segments.iter().cloned()));
    let all = || {
        h.components
            .iter()
            .flat_map(|(c, ts)| ts.iter().map(move |t| (c, t)))
    };
    for (_, t) in all() {
        for p in &t.parts {
            if let v1::Body::Objects(objects) = &p.body {
                r.v1.extend(objects.iter().cloned());
            }
        }
    }
    let mut v1_doc = json!({"manifest": object(&lane.manifest), "header": object(&header), "tensors": all().count()});
    v1_doc
        .as_object_mut()
        .expect("an object")
        .extend(count(&r.v1).as_object().expect("an object").clone());
    r.doc.insert("v1".into(), v1_doc);

    let mappings: Vec<(String, &Mapping)> = h.specs.iter().map(v1::Spec::mapping).collect();
    let (mut schemes, mut dropped) = (
        BTreeMap::<String, u64>::new(),
        BTreeMap::<String, u64>::new(),
    );
    let mut encodings = BTreeMap::<String, u64>::new();
    for (_, t) in all() {
        let (alias, mapping) = &mappings[t.spec];
        *encodings.entry(alias.clone()).or_default() += 1;
        match mapping {
            Mapping::Plain => *schemes.entry("plain".into()).or_default() += 1,
            Mapping::Scheme(s, _) => *schemes.entry(s.to_string()).or_default() += 1,
            Mapping::Dropped => *dropped.entry(alias.clone()).or_default() += 1,
        }
    }
    r.doc.insert("encodings".into(), json!(encodings));
    let mut names = Names::of(&h);
    let (order_doc, expected) = ordering(&h, &names);
    r.doc.insert("order".into(), order_doc);
    r.doc.insert("schemes".into(), json!(schemes));
    if !dropped.is_empty() {
        r.doc.insert("dropped_schemes".into(), json!(dropped));
        return Ok(false);
    }
    let mut nonconforming = BTreeMap::<String, (u64, String)>::new();
    for (c, t) in all() {
        if let Mapping::Scheme(scheme, _) = mappings[t.spec].1 {
            if let Err(why) = conforms(t, scheme, &v2_parts(t, mappings[t.spec].1)?) {
                let e = nonconforming.entry(mappings[t.spec].0.clone()).or_default();
                e.0 += 1;
                if e.1.is_empty() {
                    e.1 = format!("{c}/{}: {why}", t.key);
                }
            }
        }
    }
    if !nonconforming.is_empty() {
        r.doc.insert("nonconforming".into(), json!(nonconforming));
        return Ok(false);
    }

    let files = files(&m, &h, &mut names)?;
    if let Some(doc) = names.report() {
        r.doc.insert("renames".into(), doc);
    }
    let Some(store) = store else {
        let (mut packed, mut reused) = (0u64, 0u64);
        let mut new = lane.manifest.length + header.length;
        new += h.configs.iter().map(|c| c.1.len() as u64).sum::<u64>();
        for p in all().flat_map(|(_, t)| &t.parts) {
            match reusable(p) {
                Some(_) => reused += 1,
                None => (packed, new) = (packed + 1, new + p.length().next_multiple_of(ALIGN)),
            }
        }
        let plan = json!({"new_bytes": new, "packed_parts": packed, "reused_parts": reused});
        r.doc.insert("plan".into(), plan);
        return Ok(true);
    };
    let key = format!("migrate-{}", lane.manifest.sha256);
    let mut w = store.writer(None, Some(&key))?;
    let kept: Vec<(&v1::Tensor, String, String)> = (expected.iter())
        .map(|(c, i)| {
            let (component, tensors) = &h.components[*c];
            let t = &tensors[*i];
            (t, names.component(component), names.key(component, &t.key))
        })
        .collect();
    let order: Vec<(&str, &str)> = kept
        .iter()
        .map(|(_, c2, k2)| (c2.as_str(), k2.as_str()))
        .collect();
    w.set_order(&order);
    let mut digests = HashMap::new();
    for (t, c2, k2) in &kept {
        if !w.has(c2, k2) {
            put(&mut w, src, c2, k2, t, mappings[t.spec].1, &mut digests)?;
        }
    }
    w.set_metadata("migrated_from", Value::Str(lane.manifest.sha256.clone()));
    for (path, file) in &files {
        match file {
            File::Reuse(object) => w.reuse_file(path, object)?,
            File::Bytes(bytes) => w.file(path, bytes)?,
            File::Assemble { segments, object } => {
                let mut bytes = Vec::new();
                for s in segments {
                    bytes.extend(src.read(s)?);
                }
                if ObjectRef::of(&bytes) != *object {
                    return integrity(format!("asset {path} is not {}", object.sha256));
                }
                w.file(path, &bytes)?
            }
            File::Split { v1, component } => {
                let doc: Json =
                    serde_json::from_slice(v1).or_else(|e| fail(Kind::Invalid, e.to_string()))?;
                let bytes = serde_json_canonicalizer::to_vec(&doc[component.as_str()])
                    .or_else(|e| fail(Kind::Invalid, e.to_string()))?;
                w.file(path, &bytes)?
            }
        }
    }
    let manifest = w.commit()?;
    r.manifests.1 = Some(manifest.clone());
    let ck = store.checkpoint(&manifest)?;
    r.v2.insert(manifest.clone());
    r.v2.extend(ck.manifest().files.values().cloned());
    r.v2.extend(ck.header().chunks.iter().cloned());
    let new: BTreeSet<ObjectRef> = r.v2.difference(&r.v1).cloned().collect();
    let reused: BTreeSet<ObjectRef> = r.v2.intersection(&r.v1).cloned().collect();
    let mut v2_doc = json!({
        "manifest": object(&manifest),
        "header": object(&ck.manifest().files[HEADER_PATH]),
        "chunks": ck.header().chunks.len(),
        "files": ck.manifest().files.len() - 1,
    });
    v2_doc
        .as_object_mut()
        .expect("an object")
        .extend(count(&r.v2).as_object().expect("an object").clone());
    r.doc.extend([
        ("v2".into(), v2_doc),
        ("new".into(), count(&new)),
        ("reused".into(), count(&reused)),
    ]);
    let verified = verify(
        src, store, &manifest, &h, &names, &mappings, &digests, &files, &order,
    );
    if dry_run {
        store.unpin(&key)?;
        store.clone().with_grace(Duration::ZERO).gc()?;
    }
    r.doc.insert("verified".into(), verified?);
    Ok(true)
}

/// The order v2 writes, as `(component, tensor)` indices into the v1 header, and its report.
/// Components keep v1's order. Within one, v1's tensor order stands when v1 built it; when v1
/// sorted it (`order_class`), the component's class gives its construction order
/// ([`constructor_order`]), keys it does not know keeping v1's order. Renamed keys keep their
/// place; dropped ones leave.
fn ordering(h: &v1::Header, names: &Names) -> (Json, Vec<(usize, usize)>) {
    let mut components = Map::new();
    let (mut classes, mut out) = (BTreeSet::new(), Vec::new());
    for (c, (name, tensors)) in h.components.iter().enumerate() {
        let kept: Vec<usize> = (0..tensors.len())
            .filter(|i| {
                !names
                    .dropped
                    .contains(&(name.clone(), tensors[*i].key.clone()))
            })
            .collect();
        let v1_keys: Vec<&str> = tensors.iter().map(|t| t.key.as_str()).collect();
        let (class, evidence) = order_class(&v1_keys);
        if class != "unknown" {
            classes.insert(class);
        }
        let mut doc = json!({"class": class, "evidence": evidence});
        let mut order = kept.clone();
        if class == "sorted" {
            let keys: Vec<String> = kept
                .iter()
                .map(|i| names.key(name, &tensors[*i].key))
                .collect();
            let indexed: Vec<(&str, usize)> = keys.iter().map(String::as_str).zip(0..).collect();
            // An adapter's keys end in factors its class lacks: count its modules instead.
            let (model, unknown) = match model_class(h, name) {
                Some(model) => (Some(model), None),
                None => probed_class(&keys).map_or((None, None), |(c, n)| (Some(c), Some(n))),
            };
            let ordered = constructor_order(model.as_deref(), &indexed);
            order = ordered.order.iter().map(|i| kept[*i]).collect();
            doc["reference"] = json!(model);
            doc["unreferenced"] = unknown.unwrap_or(ordered.unknown.len()).into();
            doc["head"] = json!(ordered
                .order
                .iter()
                .take(4)
                .map(|i| &keys[*i])
                .collect::<Vec<_>>());
        }
        out.extend(order.into_iter().map(|i| (c, i)));
        components.insert(name.clone(), doc);
    }
    let class = match classes.len() {
        0 => "unknown",
        1 => classes.first().copied().unwrap_or("unknown"),
        _ => "mixed",
    };
    (json!({"class": class, "components": components}), out)
}

/// A component's model class, from v1's inline configs: its own, its entry in a `model` config
/// holding every component's, or a single component's `model` config.
fn model_class(h: &v1::Header, component: &str) -> Option<String> {
    let config = |name: &str| {
        let bytes = &h.configs.iter().find(|c| c.0 == name)?.1;
        serde_json::from_slice::<Json>(bytes).ok()
    };
    let class = |doc: &Json| {
        let name = doc
            .get("_class_name")
            .or_else(|| doc.get("architectures")?.get(0));
        name.and_then(Json::as_str).map(str::to_owned)
    };
    if let Some(doc) = config(component) {
        return class(&doc);
    }
    let model = config("model")?;
    class(&model[component]).or_else(|| (h.components.len() == 1).then(|| class(&model))?)
}

/// LoRA factors (and their PEFT adapter name) under a module of the class they adapt.
const FACTORS: [&str; 5] = ["lora_A", "lora_B", "lora_down", "lora_up", "alpha"];

/// The class whose construction order holds the modules of `keys` (an adapter carries no
/// config): the one that knows all but under a tenth of them, as `<module>.weight`, and how
/// many it does not know.
fn probed_class(keys: &[String]) -> Option<(String, usize)> {
    let probes: Vec<String> = (keys.iter())
        .map(|k| {
            let segments: Vec<&str> = k.split('.').collect();
            match segments.iter().position(|s| FACTORS.contains(s)) {
                Some(i) => format!("{}.weight", segments[..i].join(".")),
                None => k.clone(),
            }
        })
        .collect();
    let indexed: Vec<(&str, usize)> = probes.iter().map(String::as_str).zip(0..).collect();
    let (best, unknown) = order::classes()
        .map(|c| (c, constructor_order(Some(c), &indexed).unknown.len()))
        .min_by_key(|(_, unknown)| *unknown)?;
    (unknown * 10 < keys.len()).then(|| (best.to_owned(), unknown))
}

fn order_class(keys: &[&str]) -> (&'static str, String) {
    let n = keys.len();
    let descents = keys.windows(2).filter(|w| w[0] > w[1]).count();
    let at: HashMap<&str, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let (mut pairs, mut weight_first) = (0, 0);
    for (i, k) in keys.iter().enumerate() {
        let weight = k
            .strip_suffix(".bias")
            .and_then(|m| at.get(format!("{m}.weight").as_str()));
        if let Some(w) = weight {
            pairs += 1;
            weight_first += usize::from(*w < i);
        }
    }
    let why =
        format!("{descents} lexical descents in {n}; weight before bias {weight_first}/{pairs}");
    let class = match (n, descents) {
        (0..=2, _) => "unknown",
        (_, 0) => "sorted",
        (0..=19, _) => "unknown",
        _ if descents <= 3 || descents * 20 < n => "sorted",
        _ => "constructor",
    };
    let contradicted = pairs > 0
        && match class {
            "sorted" => weight_first == pairs,
            "constructor" => weight_first == 0,
            _ => false,
        };
    (if contradicted { "unknown" } else { class }, why)
}

/// D4's canonical names for one v1 checkpoint.
#[derive(Default)]
struct Names {
    components: BTreeMap<String, String>,
    /// `(v1 component, v1 key)` → v2 key, for the keys that change.
    keys: HashMap<(String, String), String>,
    dropped: BTreeSet<(String, String)>,
    rules: Vec<String>,
}

impl Names {
    fn of(h: &v1::Header) -> Names {
        let mut n = Names::default();
        for (component, tensors) in &h.components {
            let keys: Vec<&str> = tensors.iter().map(|t| t.key.as_str()).collect();
            let name = match canonical_component(component, &keys) {
                Some(name) => {
                    n.components.insert(component.clone(), name.clone());
                    name
                }
                None => component.clone(),
            };
            let Some(Fit {
                table,
                renames,
                drop,
            }) = table_keys(&name, &keys)
            else {
                continue;
            };
            if let Some((from, to)) = renames.first() {
                let rule = format!(
                    "{component}: {} keys renamed by {table}, e.g. {from} → {to}",
                    renames.len()
                );
                n.rules.push(rule);
            }
            for (from, to) in renames {
                n.keys.insert((component.clone(), from), to);
            }
            for key in drop {
                n.rules
                    .push(format!("{component}: {key} dropped by {table}"));
                n.dropped.insert((component.clone(), key));
            }
        }
        n
    }

    fn component(&self, c: &str) -> String {
        self.components
            .get(c)
            .cloned()
            .unwrap_or_else(|| c.to_owned())
    }

    fn key(&self, c: &str, k: &str) -> String {
        let renamed = self.keys.get(&(c.to_owned(), k.to_owned()));
        renamed.cloned().unwrap_or_else(|| k.to_owned())
    }

    fn report(&self) -> Option<Json> {
        if self.components.is_empty() && self.rules.is_empty() {
            return None;
        }
        Some(json!({"components": self.components, "rules": self.rules}))
    }
}

/// A v1 component's canonical name, when it differs: an H3 folder's, or a LoRA's single table
/// component that holds every module it adapts.
fn canonical_component(component: &str, keys: &[&str]) -> Option<String> {
    if let Some((_, folder)) = V1_FOLDERS.iter().find(|(v1, _)| *v1 == component) {
        let tabled =
            (tables().iter().flat_map(|t| &t.components)).find_map(|c| c.folders.get(*folder));
        return tabled
            .cloned()
            .or_else(|| (!folder.contains('/')).then(|| folder.to_string()));
    }
    let stems: BTreeSet<&str> = keys
        .iter()
        .map(|k| LORA.iter().find_map(|s| k.strip_suffix(s)))
        .collect::<Option<_>>()?;
    let mut claims = (tables().iter().filter(|t| t.lora.is_some()))
        .flat_map(|t| &t.components)
        .filter(|c| {
            let modules: BTreeSet<&str> = c.rows.iter().map(|r| r.target.as_str()).collect();
            !stems.is_empty() && stems.iter().all(|s| modules.contains(s))
        });
    match (claims.next(), claims.next()) {
        (Some(c), None) if c.name != component => Some(c.name.clone()),
        _ => None,
    }
}

/// How a v1 component's keys become a table component's canonical keys.
struct Fit {
    table: String,
    /// `(v1 key, v2 key)`.
    renames: Vec<(String, String)>,
    drop: Vec<String>,
}

/// The renames and drops that make `keys` a table component's canonical keys, when one fits.
fn table_keys(component: &str, keys: &[&str]) -> Option<Fit> {
    for t in tables().iter().filter(|t| t.lora.is_none()) {
        let named = |c: &&crate::import::engine::Component| {
            c.name == component || c.folders.values().any(|f| f == component)
        };
        for c in t.components.iter().filter(named) {
            let targets: BTreeSet<&str> = c.rows.iter().map(|r| r.target.as_str()).collect();
            let drop: Vec<String> = (keys.iter())
                .filter(|k| c.drop.iter().any(|d| d == *k))
                .map(|k| k.to_string())
                .collect();
            let missing: Vec<&str> = (keys.iter().copied())
                .filter(|k| !targets.contains(k) && !drop.iter().any(|d| d == k))
                .collect();
            // A prefix v1 left off (SDXL's `text_model.`).
            let prefix = match missing.first() {
                None => "",
                Some(m) => match targets
                    .iter()
                    .find_map(|t| t.strip_suffix(m).filter(|p| p.ends_with('.')))
                {
                    Some(p) => p,
                    None => continue,
                },
            };
            if missing
                .iter()
                .all(|k| targets.contains(format!("{prefix}{k}").as_str()))
            {
                let renames = (missing.iter())
                    .map(|k| (k.to_string(), format!("{prefix}{k}")))
                    .collect();
                return Some(Fit {
                    table: t.name.clone(),
                    renames,
                    drop,
                });
            }
        }
    }
    None
}

/// A part on v1's 64 MiB grid and large enough to stand alone in v2: its objects carry over.
fn reusable(p: &v1::Part) -> Option<&[ObjectRef]> {
    let v1::Body::Objects(objects) = &p.body else {
        return None;
    };
    let last = objects.len().checked_sub(1)?;
    let on_grid = (objects.iter().enumerate())
        .all(|(i, o)| o.length > 0 && o.length <= GRID && (i == last || o.length == GRID));
    (on_grid && p.length() >= STANDALONE).then_some(objects.as_slice())
}

fn bytes(src: &v1::Source, p: &v1::Part) -> Result<Vec<u8>> {
    match &p.body {
        v1::Body::Inline(bytes) => Ok(bytes.clone()),
        v1::Body::Objects(objects) => {
            let mut out = Vec::with_capacity(p.length() as usize);
            for o in objects {
                out.extend(src.read(o)?);
            }
            Ok(out)
        }
    }
}

/// The sha256 of a part's v1 bytes `b`: a part of one object is that object, already checked.
fn digest(p: &v1::Part, b: &[u8]) -> String {
    match &p.body {
        v1::Body::Objects(objects) if objects.len() == 1 => objects[0].sha256.clone(),
        _ => hex_digest(b),
    }
}

/// The scheme's complete encoding; `weight` is the weight part's dtype.
fn encoding(scheme: &str, weight: &Dtype) -> Encoding {
    match scheme {
        "fp8.rowwise" => Encoding::fp8_rowwise(weight),
        "fp8.tensorwise" => Encoding::fp8_tensorwise(weight),
        "int8.convrot" => Encoding::int8_convrot(256),
        "nvfp4" => Encoding::nvfp4(),
        other => unreachable!("v1 maps to no {other}"),
    }
}

/// Each v1 part's v2 role, dtype and shape, in v1 order.
fn v2_parts(t: &v1::Tensor, mapping: &Mapping) -> Result<Vec<(String, Dtype, Vec<u64>)>> {
    let what = &t.key;
    match mapping {
        Mapping::Plain => match t.parts.as_slice() {
            [p] if p.dtype == t.dtype && p.shape == t.shape => {
                Ok(vec![(String::new(), p.dtype.clone(), p.shape.clone())])
            }
            _ => fail(
                Kind::Invalid,
                format!("{what}: a plain tensor is not one part of its dtype and shape"),
            ),
        },
        Mapping::Scheme(scheme, roles) => (t.parts.iter())
            .map(|p| {
                let Some((_, role)) = roles.iter().find(|(v1, _)| *v1 == p.role) else {
                    return fail(Kind::Invalid, format!("{what}: no v2 role for {}", p.role));
                };
                match (*scheme, *role, t.shape.as_slice()) {
                    ("nvfp4", "weight", [rows, cols])
                        if p.dtype == Dtype::U8 && p.shape == [*rows, cols / 2] =>
                    {
                        // v1's packed `U8 [R, C/2]` is `F4 [R, C]`: the same bytes.
                        Ok((role.to_string(), Dtype::F4, vec![*rows, *cols]))
                    }
                    ("nvfp4", "weight", _) => {
                        fail(Kind::Invalid, format!("{what}: nvfp4 weight {:?}", p.shape))
                    }
                    _ => Ok((role.to_string(), p.dtype.clone(), p.shape.clone())),
                }
            })
            .collect(),
        Mapping::Dropped => fail(Kind::Unsupported, format!("{what}: a dropped scheme")),
    }
}

/// The mapped parts against v2's definition of `scheme`, which its decoder enforces.
fn conforms(
    t: &v1::Tensor,
    scheme: &str,
    parts: &[(String, Dtype, Vec<u64>)],
) -> std::result::Result<(), String> {
    use Dtype::*;
    let numel = |s: &[u64]| s.iter().product::<u64>();
    if !matches!(t.dtype, F32 | BF16 | F16) {
        return Err(format!("logical dtype {}", t.dtype));
    }
    let (rows, cols) = match t.shape[..] {
        [r, c] => (r, c),
        _ if scheme == "fp8.tensorwise" => (1, numel(&t.shape)),
        _ => return Err(format!("logical shape {:?} is not rank 2", t.shape)),
    };
    let want = |role: &str, dtypes: &[Dtype], ok: &dyn Fn(&[u64]) -> bool| match parts
        .iter()
        .find(|p| p.0 == role)
    {
        Some((_, d, s)) if dtypes.contains(d) && ok(s) => Ok(()),
        Some((_, d, s)) => Err(format!("part {role} is {d} {s:?}")),
        None => Err(format!("no part {role}")),
    };
    let optional = |role: &str| match parts.iter().any(|p| p.0 == role) {
        true => want(role, &[F32], &|s| numel(s) == 1),
        false => Ok(()),
    };
    let (fp8, logical) = ([F8_E4M3, F8_E5M2], t.shape.as_slice());
    match scheme {
        "fp8.rowwise" => {
            want("weight", &fp8, &|s| s == logical)?;
            want("scale", &[F32], &|s| numel(s) == rows)
        }
        "fp8.tensorwise" => {
            want("weight", &fp8, &|s| s == logical)?;
            want("scale", &[F32], &|s| numel(s) == 1)?;
            optional("input_scale")
        }
        "int8.convrot" if cols % 256 == 0 => {
            want("weight", &[I8], &|s| s == logical)?;
            want("scale", &[F32], &|s| numel(s) == rows)
        }
        "nvfp4" if cols % 16 == 0 => {
            want("weight", &[F4], &|s| s == [rows, cols])?;
            want("scale", &[F8_E4M3], &|s| s == [rows, cols / 16])?;
            want("global_scale", &[F32], &|s| numel(s) == 1)?;
            optional("input_scale")
        }
        _ => Err(format!("{cols} columns do not fit {scheme}")),
    }
}

/// `(v2 component, v2 key, v2 role)` → sha256 of the part's v1 bytes, for packed parts.
type Digests = HashMap<(String, String, String), String>;

/// One v1 tensor through the writer: large grid parts reused, the rest read and digested.
fn put(
    w: &mut Writer,
    src: &v1::Source,
    c2: &str,
    k2: &str,
    t: &v1::Tensor,
    mapping: &Mapping,
    digests: &mut Digests,
) -> Result<()> {
    let roles = v2_parts(t, mapping)?;
    let mut read = Vec::new();
    for (p, (role, ..)) in t.parts.iter().zip(&roles) {
        if reusable(p).is_some() {
            read.push(Vec::new());
            continue;
        }
        let b = bytes(src, p)?;
        digests.insert((c2.to_owned(), k2.to_owned(), role.clone()), digest(p, &b));
        read.push(b);
    }
    let mut slices: Vec<&[u8]> = read.iter().map(Vec::as_slice).collect();
    let mut parts: Vec<PartData> = Vec::new();
    for ((p, (role, dtype, shape)), slice) in t.parts.iter().zip(roles).zip(slices.iter_mut()) {
        let data = match reusable(p) {
            Some(objects) => Source::Reuse(objects.to_vec()),
            None => Source::Read(slice),
        };
        parts.push(PartData {
            role,
            dtype,
            shape,
            data,
        });
    }
    let data = match mapping {
        Mapping::Scheme(scheme, _) => Data::Encoded {
            encoding: encoding(scheme, &parts.iter().find(|p| p.role == "weight").expect("conforms() checked the weight").dtype),
            parts,
        },
        _ => Data::Plain(parts.pop().expect("a plain tensor has one part").data),
    };
    Ok(w.put(c2, k2, &t.dtype, &t.shape, data)?)
}

/// A file of the v2 tree.
enum File {
    /// A v1 file or single-object asset, by its object.
    Reuse(ObjectRef),
    /// A v1 config, as its v1 bytes.
    Bytes(Vec<u8>),
    /// A multi-object asset, joined.
    Assemble {
        segments: Vec<ObjectRef>,
        object: ObjectRef,
    },
    /// One component's config out of v1's `model` config that holds every component's (H3).
    Split { v1: Vec<u8>, component: String },
}

/// The v2 tree's files: v1's files and assets, and its inline configs at their Diffusers or
/// Transformers paths.
fn files(m: &v1::Manifest, h: &v1::Header, names: &mut Names) -> Result<BTreeMap<String, File>> {
    let mut out: BTreeMap<String, File> = BTreeMap::new();
    fn add(out: &mut BTreeMap<String, File>, path: String, file: File) -> Result<()> {
        let same = match (out.get(&path), &file) {
            (None, _) => false,
            (Some(File::Reuse(a)), File::Reuse(b)) if a == b => true,
            _ => return fail(Kind::Invalid, format!("two v1 files land at {path}")),
        };
        if !same {
            out.insert(path, file);
        }
        Ok(())
    }
    for (path, object) in &m.files {
        add(&mut out, path.clone(), File::Reuse(object.clone()))?;
    }
    for a in &h.assets {
        let file = match a.segments.as_slice() {
            [one] if *one == a.object => File::Reuse(one.clone()),
            segments => File::Assemble {
                segments: segments.to_vec(),
                object: a.object.clone(),
            },
        };
        add(&mut out, a.path.clone(), file)?;
    }
    let components: BTreeSet<&str> = h.components.iter().map(|c| c.0.as_str()).collect();
    for (name, bytes) in &h.configs {
        let path = match name.as_str() {
            "model_index" => "model_index.json".to_owned(),
            "scheduler" => "scheduler/scheduler_config.json".to_owned(),
            "model" => {
                let doc: Json = serde_json::from_slice(bytes).unwrap_or_default();
                let split = doc.as_object().filter(|map| {
                    !components.contains("model")
                        && map.len() == components.len()
                        && map.keys().all(|k| components.contains(k.as_str()))
                });
                if let Some(map) = split {
                    for c in map.keys() {
                        let path = format!("{}/config.json", names.component(c));
                        add(
                            &mut out,
                            path,
                            File::Split {
                                v1: bytes.clone(),
                                component: c.clone(),
                            },
                        )?;
                    }
                    names
                        .rules
                        .push("config model: one config.json per component".into());
                    continue;
                }
                "config.json".to_owned()
            }
            c if components.contains(c) => format!("{}/config.json", names.component(c)),
            other => format!("{other}.json"),
        };
        if matches!(out.get(&path), Some(File::Reuse(_))) {
            names
                .rules
                .push(format!("config {name}: the tree's own {path} kept"));
            continue;
        }
        add(&mut out, path, File::Bytes(bytes.clone()))?;
    }
    Ok(out)
}

/// The committed v2 checkpoint against v1, tensor by tensor and file by file.
#[allow(clippy::too_many_arguments)]
fn verify(
    src: &v1::Source,
    store: &Store,
    manifest: &ObjectRef,
    h: &v1::Header,
    names: &Names,
    mappings: &[(String, &Mapping)],
    digests: &Digests,
    files: &BTreeMap<String, File>,
    expected: &[(&str, &str)],
) -> Result<Json> {
    let ck = store.checkpoint(manifest)?;
    let header = ck.header();
    let (mut tensors, mut reused, mut packed, mut packed_bytes) = (0usize, 0u64, 0u64, 0u64);
    let mut misordered_parts = None;
    for (component, ts) in &h.components {
        for t in ts {
            if names.dropped.contains(&(component.clone(), t.key.clone())) {
                continue;
            }
            let (c2, k2) = (names.component(component), names.key(component, &t.key));
            let what = format!("{component}/{} (v2 {c2}/{k2})", t.key);
            let t2 = ck.tensor(&c2, &k2)?;
            if t2.dtype != t.dtype || t2.shape != t.shape {
                return integrity(format!("{what}: {} {:?} in v2", t2.dtype, t2.shape));
            }
            let mapping = mappings[t.spec].1;
            let stored: Vec<(&str, &Dtype, &[u64], Location)> = match (&t2.body, mapping) {
                (Body::Data(at), Mapping::Plain) => vec![("", &t2.dtype, &t2.shape, *at)],
                (Body::Encoded { encoding: e, parts }, Mapping::Scheme(s, _))
                    if parts.len() == t.parts.len()
                        && (parts.iter().find(|p| p.role == "weight"))
                            .is_some_and(|w| *e == encoding(s, &w.dtype)) =>
                {
                    (parts.iter())
                        .map(|p| (p.role.as_str(), &p.dtype, p.shape.as_slice(), p.at()))
                        .collect()
                }
                _ => return integrity(format!("{what}: its v2 body or scheme differs")),
            };
            let roles = v2_parts(t, mapping)?;
            if !stored
                .iter()
                .map(|s| s.0)
                .eq(roles.iter().map(|r| r.0.as_str()))
            {
                misordered_parts = misordered_parts.or(Some(what.clone()));
            }
            for (p, (role, dtype, shape)) in t.parts.iter().zip(roles) {
                let Some((_, d2, s2, at)) = stored.iter().find(|s| s.0 == role) else {
                    return integrity(format!("{what}: no v2 part {role:?}"));
                };
                if **d2 != dtype || *s2 != shape.as_slice() {
                    return integrity(format!("{what} part {role:?}: {d2} {s2:?} in v2"));
                }
                match reusable(p) {
                    Some(objects) => {
                        let spans = header.spans(at);
                        let same = at.offset == 0
                            && spans.len() == objects.len()
                            && (spans.iter().zip(objects)).all(|(s, o)| {
                                s.start == 0 && s.length == o.length && header.chunks[s.chunk] == *o
                            });
                        if !same {
                            return integrity(format!("{what} part {role:?}: not v1's objects"));
                        }
                        reused += 1;
                    }
                    None => {
                        let mut buf = vec![0; at.length as usize];
                        ck.bytes_into(at, &mut buf)?;
                        let want = match digests.get(&(c2.clone(), k2.clone(), role.clone())) {
                            Some(d) => d.clone(),
                            None => digest(p, &bytes(src, p)?),
                        };
                        if hex_digest(&buf) != want {
                            return integrity(format!(
                                "{what} part {role:?}: bytes differ from v1"
                            ));
                        }
                        packed += 1;
                        packed_bytes += at.length;
                    }
                }
            }
            tensors += 1;
        }
    }
    if ck.tensors().count() != tensors {
        return integrity(format!(
            "v2 has {} tensors, v1 {tensors}",
            ck.tensors().count()
        ));
    }
    // Order is data: v2 lists tensors in the expected order and parts in v1's.
    let stored = ck.tensors().map(|(c, t)| (c, t.key.as_str()));
    if let Some((i, (v2, want))) = stored.zip(expected).enumerate().find(|(_, (a, b))| a != *b) {
        return integrity(format!(
            "order: tensor {i} is {}/{} in v2, {}/{} expected",
            v2.0, v2.1, want.0, want.1
        ));
    }
    if let Some(what) = misordered_parts {
        return integrity(format!("order: {what}'s parts are not in v1's order"));
    }
    let mut stored = ck.manifest().files.clone();
    stored.remove(HEADER_PATH);
    if !stored.keys().eq(files.keys()) {
        return integrity("the v2 files are not v1's");
    }
    for (path, file) in files {
        let same = match file {
            File::Reuse(object) => stored[path] == *object,
            File::Bytes(bytes) => ck.file(path)? == *bytes,
            File::Assemble { object, .. } => stored[path] == *object,
            File::Split { v1, component } => {
                let parse = |b: &[u8]| serde_json::from_slice::<Json>(b).unwrap_or_default();
                let v2 = parse(&ck.file(path)?);
                !v2.is_null() && v2 == parse(v1)[component.as_str()]
            }
        };
        if !same {
            return integrity(format!("file {path} differs from v1"));
        }
    }
    Ok(json!({
        "tensors": tensors,
        "reused_parts": reused,
        "packed_parts": packed,
        "packed_bytes": packed_bytes,
        "files": files.len(),
    }))
}
