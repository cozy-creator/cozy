//! xs-053 A11: models from before D10 become D10, a cozytensors per lane and a tree per release.
//! A lane is a v1 checkpoint (production's; [`super::v1`] reads it) or a 0.12 one (a
//! `tensorfs.manifest/1` and a five-field header). Header-only: every chunk, tensor, part and
//! location stays where it was. Run data a lane lacks comes from its family's pinned upstream
//! ([`UPSTREAMS`], owner Decision 1) and is recorded in the header's `filled`; every lane records
//! its `source`, `unknown` when nobody did. Deleted once it has run.

use std::collections::{btree_map::Entry, BTreeMap, HashMap};
use std::fs;
use std::io::Read as _;
use std::path::Path;
use std::sync::Arc;

use base64::Engine as _;
use bytes::Bytes;
use flate2::read::ZlibDecoder;
use minicbor::Decoder;
use serde::Deserialize;
use serde_json::{json, Value as Json};

use crate::failure::{fail, Failure, Kind, Result};
use crate::fetch::{self, Client, Endpoint, Fetcher, Fresh, Origin, Payload, SourcePolicy, Want};
use crate::format::{Component, Config, Header, Metadata, ObjectRef, Release, Repo, Tensor, Tree, Value, HEADER_PATH};
use crate::import::{layout, remote};
use crate::store::Store;

use super::v1;

/// A family's pinned upstream and what it fills where a lane has none: a path, a folder
/// (`tokenizer/`), or `README`/`LICENSE` (any top-level file so named).
pub struct Upstream {
    /// `org/name` patterns, `*` any run of characters.
    pub models: &'static [&'static str],
    /// The pinned upstream the files come from; `""` when nothing is filled.
    pub from: &'static str,
    pub files: &'static [&'static str],
    /// The weights' source, when it is not `from`: `civitai://<version>`, another pin, `local`.
    pub source: Option<&'static str>,
    pub note: &'static str,
}

const NOTE: &str = "run data the lane lacked, from its family's upstream at a pinned commit";
const SDXL: &str = "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b";
const ANIMA: &str = "hf://circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9db359c31ad0e8aa268d15775473c2176c";
const QWEN_IMAGE: &str = "hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758";
const H3: &str = "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe";
/// What a civitai SDXL fine-tune lacks: SDXL's pipeline and both tokenizers, never its docs.
const PIPELINE: [&str; 3] = ["model_index.json", "tokenizer/", "tokenizer_2/"];
const ANIMA_RUN: [&str; 3] = ["modular_model_index.json", "tokenizer/", "t5_tokenizer/"];
/// SDXL base's own lanes take any run file they lack from the base, configs included: the lanes'
/// configs are the base's.
const SDXL_RUN: [&str; 10] = [
    "model_index.json", "scheduler/", "tokenizer/", "tokenizer_2/", "text_encoder/config.json",
    "text_encoder_2/config.json", "unet/config.json", "vae/config.json", "README", "LICENSE",
];
const DOCS: [&str; 2] = ["README", "LICENSE"];

const fn row(models: &'static [&'static str], from: &'static str, files: &'static [&'static str]) -> Upstream {
    Upstream { models, from, files, source: None, note: NOTE }
}

const fn civitai(models: &'static [&'static str], version: &'static str) -> Upstream {
    Upstream { models, from: SDXL, files: &PIPELINE, source: Some(version), note: NOTE }
}

/// The owner-approved table (2026-10-19, civitai and source rulings 2026-10-20), each entry
/// checked against its pin. Civitai fine-tunes get run data only, never README or LICENSE.
pub const UPSTREAMS: &[Upstream] = &[
    row(&["*/sdxl", "*/sdxl-fp8-postcut-proof", "paul/d1-*sdxl*"], SDXL, &SDXL_RUN),
    civitai(&["*/babes-pony"], "civitai://2937668"),
    civitai(&["*/big-love-xl"], "civitai://2857189"),
    civitai(&["*/cyberrealistic-pony"], "civitai://2884631"),
    civitai(&["*/cyberrealistic-xl"], "civitai://2840768"),
    civitai(&["*/diving-illustrious-anime"], "civitai://2939886"),
    civitai(&["*/furrytoonmix-xl"], "civitai://2961728"),
    civitai(&["*/gonzalomo-xl"], "civitai://2951793"),
    civitai(&["*/nova-anime-xl"], "civitai://2940478"),
    civitai(&["*/nova-furry-xl"], "civitai://2943166"),
    civitai(&["*/nova-orange-xl"], "civitai://2945776"),
    civitai(&["*/perfectdeliberate-anime"], "civitai://2925672"),
    civitai(&["*/pie-models-mango"], "civitai://2913030"),
    civitai(&["*/prefect-illustrious-xl"], "civitai://2864868"),
    civitai(&["*/realism-illustrious", "*/civitai-2831949"], "civitai://2831949"),
    civitai(&["*/wai-illustrious"], "civitai://2883731"),
    row(&["*/anima"], ANIMA, &["modular_model_index.json", "tokenizer/", "t5_tokenizer/", "README", "LICENSE"]),
    Upstream { models: &["*/anima-civitai"], from: ANIMA, files: &ANIMA_RUN, source: Some("civitai://2945208"), note: NOTE },
    row(&["*/flux-3-klein"], "hf://black-forest-labs/flux-3-image-klein@8454017f9ebaada30c998fccd9f67a80c7580f68", &DOCS),
    row(&["*/flux-3-klein-base"], "hf://black-forest-labs/flux-3-image-klein-base@1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4", &DOCS),
    row(
        &["*/flux-3-klein-upsampler"],
        "hf://black-forest-labs/flux-3-image-klein-pup@8b069751eb5dcab23829d9c5eccb470eb8b8a4ed",
        &["system_prompts.json", "README"],
    ),
    row(&["*/qwen-image-2.1", "*/reference-image"], QWEN_IMAGE, &["model_index.json", "scheduler/", "processor/", "README", "LICENSE"]),
    row(&["*/qwen3.5-9b", "*/qwen3.5-9b-mtp"], "hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a", &["merges.txt", "vocab.json", "README", "LICENSE"]),
    row(&["*/minimax-h3", "*/minimax-h3-vsa-gate", "*/minimax-h3-turbo-exp", "*/minimax-h3-turbo-lora"], H3, &DOCS),
    Upstream {
        models: &["*/minimax-h3-lightx2v", "*/minimax-h3-lightx2v-source"],
        from: "hf://lightx2v/Minimax-h3-Turbo@9bc1f3c0407805b00da9f762639cb435c644b2dc",
        files: &["README"],
        source: None,
        note: "README from lightx2v's upstream, pinned at fill time, not ingest",
    },
    row(&["*/minimax-h3-acc-loras"], "hf://alibaba-pai/MiniMax-H3-Acc-LoRAs@335001fb9e5455d68a0caa18ec2e319072150328", &["README"]),
    Upstream {
        models: &["*/seedvr2-7b"],
        from: "hf://ByteDance-Seed/SeedVR2-7B@eb0c4281d41ba3767d4f14370f0e37e9e9180c16",
        files: &["README"],
        source: Some("hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8"),
        note: NOTE,
    },
    row(&["*/birefnet"], "hf://ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", &["README"]),
    // Cozy's own test fixtures and outputs: nothing to fill, made here.
    Upstream {
        models: &[
            "paul/attention-kernel-benchmark-*", "paul/cache-persistence-proof", "paul/cozy-mixed-input-proof",
            "paul/creator-output-*", "paul/hardcut-upload-proof-*", "paul/h3-cp-producer-lifecycle-*",
            "paul/h3-vae-pipeline-check", "paul/xs040-*",
        ],
        from: "",
        files: &[],
        source: Some(LOCAL),
        note: "",
    },
];

/// Cozy made it here; no upstream to name.
const LOCAL: &str = "local";
/// A lane whose source nobody recorded (owner, 2026-10-20): converted, never filled.
pub const UNKNOWN: &str = "unknown";

/// Old metadata D10's loading metadata drops; `sources` becomes `source`.
const DROPPED: [&str; 4] = ["normalized", "converters", "sources", "migrated_from"];
const CONFIG: &str = "config.json";

fn matches(pattern: &str, name: &str) -> bool {
    let (pattern, name) = (pattern.to_ascii_lowercase(), name.to_ascii_lowercase());
    let parts: Vec<&str> = pattern.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if parts.len() == 1 {
        return name == first;
    }
    let Some(mut rest) = name.strip_prefix(first) else { return false };
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(i) => rest = &rest[i + part.len()..],
            None => return false,
        }
    }
    rest.ends_with(last)
}

/// Whether `path` is under `entry`: a folder's files (`tokenizer/`), that one file
/// (`unet/config.json`), or a top-level file it begins (`README`).
fn under(entry: &str, path: &str) -> bool {
    let (entry, path) = (entry.to_ascii_lowercase(), path.to_ascii_lowercase());
    match (entry.ends_with('/'), entry.contains('/')) {
        (true, _) => path.starts_with(&entry),
        (false, true) => path == entry,
        (false, false) => !path.contains('/') && path.starts_with(&entry),
    }
}

#[derive(Deserialize)]
struct OldRepo {
    name: String,
    #[serde(default)]
    releases: Vec<OldRelease>,
    #[serde(default)]
    checkpoints: Vec<ObjectRef>,
}

#[derive(Deserialize)]
struct OldRelease {
    version: String,
    lanes: Option<BTreeMap<String, ObjectRef>>,
    tree: Option<ObjectRef>,
    #[serde(default)]
    yanked: bool,
}

#[derive(Deserialize)]
struct Doc {
    format: String,
    #[serde(default)]
    files: BTreeMap<String, ObjectRef>,
}

/// One repo in D10.
pub struct Migrated {
    pub repo: Repo,
    /// Every object the D10 repo cites besides chunks, each before what cites it: assets and
    /// repo files, headers, trees.
    pub objects: Objects,
    pub report: Json,
}

/// A lane in D10: its header, repo files and report.
struct Lane {
    header: ObjectRef,
    files: BTreeMap<String, ObjectRef>,
    report: Json,
}

pub type Read<'a> = &'a dyn Fn(&ObjectRef) -> Result<Vec<u8>>;
pub type Objects = Vec<(ObjectRef, Vec<u8>)>;

fn add(objects: &mut Objects, bytes: Vec<u8>) -> ObjectRef {
    let at = ObjectRef::of(&bytes);
    if !objects.iter().any(|(o, _)| *o == at) {
        objects.push((at.clone(), bytes));
    }
    at
}

pub struct Migrator<'t> {
    access: remote::Access,
    table: &'t [Upstream],
    listings: HashMap<String, Vec<remote::Member>>,
    fetched: HashMap<String, Vec<u8>>,
}

impl<'t> Migrator<'t> {
    pub fn new(access: remote::Access, table: &'t [Upstream]) -> Migrator<'t> {
        Migrator { access, table, listings: HashMap::new(), fetched: HashMap::new() }
    }

    /// The repo document `doc` in D10, its old objects read by `read`. A release that names a
    /// tree, and a checkpoint that is one, stay as they are; each release of lanes becomes one
    /// tree (`model.<lane>.cozytensors` each, repo files merged, the first lane's kept where
    /// they differ), yanked or not; each other old checkpoint a tree of its own
    /// (`model.cozytensors`).
    pub fn repo(&mut self, doc: &[u8], read: Read) -> Result<Migrated> {
        let old: OldRepo = serde_json::from_slice(doc).map_err(|e| invalid(format!("repo document: {e}")))?;
        let upstream = self.table.iter().find(|u| u.models.iter().any(|p| matches(p, &old.name)));
        let mut objects = Objects::new();
        let mut lanes: HashMap<ObjectRef, Lane> = HashMap::new();
        let put = |objects: &mut Objects, tree: &Tree| -> Result<ObjectRef> { Ok(add(objects, tree.to_json()?)) };
        let (mut releases, mut reports) = (Vec::new(), Vec::new());
        for r in &old.releases {
            let named = match (&r.tree, &r.lanes) {
                (Some(tree), _) => {
                    releases.push(Release { version: r.version.clone(), tree: tree.clone(), yanked: r.yanked });
                    continue;
                }
                (None, Some(named)) => named,
                (None, None) => return fail(Kind::Invalid, format!("release {} names no lanes or tree", r.version)),
            };
            let (mut tree, mut report, mut conflicts) = (Tree::default(), serde_json::Map::new(), Vec::new());
            for (name, manifest) in named {
                if !lanes.contains_key(manifest) {
                    let lane = self.lane(manifest, upstream, read, &mut objects)?;
                    lanes.insert(manifest.clone(), lane);
                }
                let lane = &lanes[manifest];
                tree.set_variant(name, lane.header.clone());
                for (path, file) in &lane.files {
                    match tree.files.get(path) {
                        Some(kept) if kept != file => conflicts.push(json!({"path": path, "dropped": name, "object": file.id()})),
                        Some(_) => {}
                        None => drop(tree.files.insert(path.clone(), file.clone())),
                    }
                }
                report.insert(name.clone(), lane.report.clone());
            }
            let release = Release { version: r.version.clone(), tree: put(&mut objects, &tree)?, yanked: r.yanked };
            reports.push(json!({"version": release.version, "tree": release.tree.id(), "yanked": release.yanked,
                "lanes": report, "conflicts": conflicts}));
            releases.push(release);
        }
        let cited: Vec<&ObjectRef> = old.releases.iter().flat_map(|r| r.lanes.iter().flat_map(|l| l.values())).collect();
        let (mut checkpoints, mut unreleased) = (Vec::new(), Vec::new());
        for c in old.checkpoints.iter().filter(|c| !cited.contains(c)) {
            let doc: Json = serde_json::from_slice(&read(c)?).map_err(|e| invalid(format!("{}: {e}", c.id())))?;
            if doc["format"].as_str().is_some_and(|f| f.starts_with("cozytensors.tree/")) {
                checkpoints.push(c.clone());
                continue;
            }
            let lane = self.lane(c, upstream, read, &mut objects)?;
            let mut tree = Tree { files: lane.files.clone() };
            tree.set_variant("", lane.header.clone());
            let at = put(&mut objects, &tree)?;
            unreleased.push(json!({"old": c.id(), "tree": at.id(), "lane": lane.report}));
            // Old checkpoints differing only in metadata D10 drops become one tree, named once.
            if !checkpoints.contains(&at) {
                checkpoints.push(at);
            }
        }
        let repo = Repo { name: old.name.clone(), releases, checkpoints };
        let report = json!({"model": old.name, "upstream": upstream.map(|u| u.from), "releases": reports, "checkpoints": unreleased});
        Ok(Migrated { repo, objects, report })
    }

    /// One old lane: its header in D10 (old tensors, filled run data, configs, assets) and its
    /// repo files; every lane file is checked to have landed in one of them.
    fn lane(&mut self, manifest: &ObjectRef, upstream: Option<&Upstream>, read: Read, objects: &mut Objects) -> Result<Lane> {
        let raw = read(manifest)?;
        let (chunks, tensors, mut metadata, mut bytes, v1_report) = match v1::manifest(&raw)? {
            Some(m) => {
                let v = v1::lane(&m, &read(&m.header)?)?;
                let mut bytes = BTreeMap::new();
                for (path, file) in v.files {
                    let b = match file {
                        v1::File::Object(o) => read(&o)?,
                        v1::File::Bytes(b) => b,
                    };
                    bytes.insert(path, b);
                }
                if let Some((_, pack)) = v.pack {
                    add(objects, pack);
                }
                (v.chunks, v.components, Metadata::new(), bytes, Some(v.report))
            }
            None => {
                let doc: Doc = serde_json::from_slice(&raw).map_err(|e| invalid(format!("{}: {e}", manifest.id())))?;
                if !doc.format.starts_with("tensorfs.manifest/") {
                    return fail(Kind::Invalid, format!("{} is a {}, not a manifest", manifest.id(), doc.format));
                }
                let mut files = doc.files;
                let Some(header) = files.remove(HEADER_PATH) else {
                    return fail(Kind::Invalid, format!("{} names no header", manifest.id()));
                };
                let old = upgrade(&read(&header)?)?;
                let mut bytes = BTreeMap::new();
                for (path, object) in &files {
                    bytes.insert(path.clone(), read(object)?);
                }
                let tensors = old.components().iter().map(|c| (c.name.clone(), c.tensors.clone())).collect();
                (old.chunks.clone(), tensors, old.metadata.clone(), bytes, None)
            }
        };
        let (unwrapped, provenance) = unwrap(&mut bytes).map_err(|e| invalid(format!("{}: {}", manifest.id(), e.detail)))?;
        let mut filled = Vec::new();
        for (u, entry) in upstream.iter().flat_map(|u| u.files.iter().map(move |e| (u, e))) {
            for (path, b) in self.upstream(u.from, entry)? {
                if let Entry::Vacant(file) = bytes.entry(path.clone()) {
                    filled.push(path);
                    file.insert(b);
                }
            }
        }
        let tensored: Vec<String> = tensors.iter().map(|(name, _)| name.clone()).collect();
        let l = layout(bytes.clone(), &tensored, &Default::default())?;
        let mut keep = |b: Vec<u8>| add(objects, b);
        let components = (l.components.into_iter())
            .map(|f| Component {
                tensors: tensors.iter().find(|(name, _)| *name == f.name).map(|(_, t)| t.clone()).unwrap_or_default(),
                assets: f.assets.into_iter().map(|(p, b)| (p, keep(b))).collect(),
                name: f.name,
                class: f.class,
                config: f.config,
                config_file: f.config_file,
            })
            .collect();
        let repo_files: BTreeMap<String, ObjectRef> = l.files.into_iter().map(|(p, b)| (p, keep(b))).collect();
        let sources = metadata.remove("sources");
        DROPPED.iter().for_each(|k| drop(metadata.remove(*k)));
        let recorded = match (provenance, sources) {
            (Some(p), _) => Some(Value::Str(p)),
            (None, Some(Value::List(mut s))) if s.len() == 1 => Some(s.remove(0)),
            (None, s) => s,
        };
        let listed = upstream.map(|u| u.source.unwrap_or(u.from)).filter(|s| !s.is_empty());
        let source = listed.map(|s| Value::Str(s.to_owned())).or(recorded).unwrap_or_else(|| Value::Str(UNKNOWN.into()));
        if let Value::Str(claim) = &source {
            self.verify(claim).map_err(|e| invalid(format!("{}: source {claim} does not verify: {}", manifest.id(), e.detail)))?;
        }
        metadata.insert("source".into(), source);
        if let (Some(u), false) = (upstream, filled.is_empty()) {
            let text = |s: &str| Value::Str(s.to_owned());
            metadata.insert("filled".into(), Value::Map(BTreeMap::from([
                ("from".to_owned(), text(u.from)),
                ("files".to_owned(), Value::List(filled.iter().map(|f| text(f)).collect())),
                ("note".to_owned(), text(u.note)),
            ])));
        }
        let new = Header::new(chunks.clone(), l.pipeline, components, metadata)?;
        let cbor = new.to_cbor()?;
        check(&chunks, &tensors, &new, &cbor, &bytes, &repo_files)?;
        let at = add(objects, cbor);
        let mut report = json!({"old": manifest.id(), "header": at.id()});
        if !filled.is_empty() {
            report["filled"] = json!(filled);
        }
        if !unwrapped.is_empty() {
            report["unwrapped"] = json!(unwrapped);
        }
        if let Some(v1) = v1_report {
            report["from_v1"] = v1;
        }
        Ok(Lane { header: at, files: repo_files, report })
    }

    /// A claimed source resolves where it says (`hf://…@commit`, `civitai://<version>`):
    /// listed once. `local` and `unknown` are markers, not claims.
    fn verify(&mut self, source: &str) -> Result<()> {
        if [LOCAL, UNKNOWN].contains(&source) || self.listings.contains_key(source) {
            return Ok(());
        }
        let listing = remote::list(&remote::parse(source)?, &self.access)?;
        self.listings.insert(source.to_owned(), listing.members);
        Ok(())
    }

    /// The files under `entry` at the pinned upstream `from`, by path.
    fn upstream(&mut self, from: &str, entry: &str) -> Result<Vec<(String, Vec<u8>)>> {
        self.verify(from)?;
        let members: Vec<remote::Member> = self.listings[from].iter().filter(|m| under(entry, &m.path)).cloned().collect();
        let mut out = Vec::new();
        for m in members {
            let key = format!("{from}/{}", m.path);
            if !self.fetched.contains_key(&key) {
                self.fetched.insert(key.clone(), remote::read_small(&m, &self.access)?);
            }
            out.push((m.path, self.fetched[&key].clone()));
        }
        Ok(out)
    }
}

fn invalid(detail: String) -> Failure {
    Failure { kind: Kind::Invalid, detail }
}

/// Cozy's native config wrappers become the files they hold, every key accounted for:
/// `config.json` as `{files: {name: zlib, base64}, model_config}` (a Transformers folder), or
/// as component configs beside `processor` files and `provenance` (Qwen-Image's first layout).
/// Returns the files it wrote and the wrapper's recorded source.
fn unwrap(bytes: &mut BTreeMap<String, Vec<u8>>) -> Result<(Vec<String>, Option<String>)> {
    let Some(Ok(Json::Object(mut w))) = bytes.get(CONFIG).map(|b| serde_json::from_slice::<Json>(b)) else {
        return Ok((vec![], None));
    };
    let pretty = |v: &Json| serde_json::to_vec_pretty(v).expect("JSON");
    let (mut out, mut source) = (Vec::new(), None);
    let transformers = w.len() == 2 && w.get("files").is_some_and(Json::is_object) && w.contains_key("model_config");
    if transformers {
        for (name, value) in w["files"].as_object().expect("an object") {
            out.push((name.clone(), inflate(name, value)?));
        }
        out.push((CONFIG.to_owned(), pretty(&w["model_config"])));
    } else if let Some(Json::Object(p)) = w.remove("provenance") {
        let text = |k: &str| p.get(k).and_then(Json::as_str).unwrap_or_default().to_owned();
        source = Some(format!("hf://{}@{}", text("source_repository"), text("source_revision")));
        for (name, value) in w {
            match (name.as_str(), &value) {
                ("processor", Json::Object(files)) => {
                    for (key, v) in files {
                        let (file, bytes) = match (key.as_str(), v) {
                            ("chat_template", Json::String(t)) => ("chat_template.jinja", t.clone().into_bytes()),
                            ("tokenizer", _) => ("tokenizer.json", inflate(key, v)?),
                            ("tokenizer_config", _) => ("tokenizer_config.json", pretty(v)),
                            ("image_processor", _) => ("preprocessor_config.json", pretty(v)),
                            ("video_processor", _) => ("video_preprocessor_config.json", pretty(v)),
                            _ => return fail(Kind::Invalid, format!("config wrapper: processor {key}")),
                        };
                        out.push((format!("processor/{file}"), bytes));
                    }
                }
                ("scheduler", Json::Object(_)) => out.push(("scheduler/scheduler_config.json".into(), pretty(&value))),
                (_, Json::Object(_)) => out.push((format!("{name}/{CONFIG}"), pretty(&value))),
                _ => return fail(Kind::Invalid, format!("config wrapper: {name}")),
            }
        }
    } else {
        return Ok((vec![], None));
    }
    bytes.remove(CONFIG);
    let mut paths = Vec::new();
    for (path, b) in out {
        if bytes.insert(path.clone(), b).is_some() {
            return fail(Kind::Invalid, format!("config wrapper and lane both hold {path}"));
        }
        paths.push(path);
    }
    Ok((paths, source))
}

/// A wrapped file: zlib, then URL-safe base64.
fn inflate(name: &str, value: &Json) -> Result<Vec<u8>> {
    let bad = |e: String| invalid(format!("config wrapper {name}: {e}"));
    let text = value.as_str().ok_or_else(|| bad("not a string".into()))?;
    let packed = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(text.trim_end_matches('=')).map_err(|e| bad(e.to_string()))?;
    let mut out = Vec::new();
    ZlibDecoder::new(&packed[..]).read_to_end(&mut out).map_err(|e| bad(e.to_string()))?;
    Ok(out)
}

/// The D10 header against the old lane's chunks and tensors and its files: the same chunks and
/// tensors, its bytes reading back to it, and each file a config, an asset or a repo file.
fn check(chunks: &[ObjectRef], tensors: &[(String, Vec<Tensor>)], new: &Header, cbor: &[u8], files: &BTreeMap<String, Vec<u8>>, repo: &BTreeMap<String, ObjectRef>) -> Result<()> {
    let integrity = |what: String| fail(Kind::Integrity, what);
    if new.chunks != chunks || Header::from_cbor(cbor)? != *new {
        return integrity("the chunks moved, or the header does not read back".into());
    }
    if let Some((c, t)) = (tensors.iter()).flat_map(|(c, ts)| ts.iter().map(move |t| (c, t))).find(|(c, t)| new.tensor(c, &t.key) != Some(*t)) {
        return integrity(format!("{c}/{} changed", t.key));
    }
    let configs: Vec<&Config> = new.components().iter().map(|c| &c.config).chain(new.pipeline().map(|p| &p.config)).collect();
    let held: Vec<&ObjectRef> = new.assets().map(|(_, _, o)| o).chain(repo.values()).collect();
    for (path, bytes) in files {
        let parsed = serde_json::from_slice::<Config>(bytes).ok();
        if !held.contains(&&ObjectRef::of(bytes)) && !parsed.is_some_and(|p| configs.contains(&&p)) {
            return integrity(format!("{path} is in neither the header nor the tree"));
        }
    }
    Ok(())
}

/// A five-field header in D10's layout, every byte kept: no pipeline, each component
/// `[name, "", {}, tensors, {}]`. D10's reader then reads it.
fn upgrade(old: &[u8]) -> Result<Header> {
    let bad = |e: minicbor::decode::Error| invalid(format!("old header: {e}"));
    let mut d = Decoder::new(old);
    let mut out = vec![0x86];
    let copy = |d: &mut Decoder, out: &mut Vec<u8>| -> Result<()> {
        let at = d.position();
        d.skip().map_err(bad)?;
        out.extend_from_slice(&old[at..d.position()]);
        Ok(())
    };
    if d.array().map_err(bad)? != Some(5) {
        return fail(Kind::Invalid, "the old header is not five fields");
    }
    for _ in 0..3 {
        copy(&mut d, &mut out)?;
    }
    out.push(0x80);
    let n = d.array().map_err(bad)?.ok_or_else(|| invalid("an indefinite components array".into()))?;
    array(&mut out, n);
    for _ in 0..n {
        if d.array().map_err(bad)? != Some(2) {
            return fail(Kind::Invalid, "an old component is not [name, tensors]");
        }
        out.push(0x85);
        copy(&mut d, &mut out)?;
        out.extend([0x60, 0xa0]);
        copy(&mut d, &mut out)?;
        out.push(0xa0);
    }
    copy(&mut d, &mut out)?;
    if d.position() != old.len() {
        return fail(Kind::Invalid, "bytes after the old header");
    }
    Ok(Header::from_cbor(&out)?)
}

/// A CBOR array head of `n` items, in its shortest form.
fn array(out: &mut Vec<u8>, n: u64) {
    let wide = n.to_be_bytes();
    match n {
        0..=23 => out.push(0x80 | n as u8),
        24..=0xff => out.extend([0x98, n as u8]),
        0x100..=0xffff => out.extend([&[0x99][..], &wide[6..]].concat()),
        0x1_0000..=0xffff_ffff => out.extend([&[0x9a][..], &wide[4..]].concat()),
        _ => out.extend([&[0x9b][..], &wide[..]].concat()),
    }
}

/// What [`migrate`] did, and the repo document as it first read it: the rollback is that
/// document put back in the endpoint's storage (a D10 Hub refuses a release going back to lanes).
pub struct Done {
    pub report: Json,
    pub before: Vec<u8>,
}

/// Repo `name` at `endpoint` in D10: the old objects it cites fetched into `scratch` over
/// `client`, what the D10 document adds uploaded where missing, then the document replaced while
/// it is as read (read and migrated again when another writer won).
pub fn migrate(endpoint: &Arc<Endpoint>, client: &Client, name: &str, scratch: &Store, migrator: &mut Migrator) -> Result<Done> {
    let rt = fetch::runtime()?;
    let (mut first, mut refused) = (None, 0);
    loop {
        let Fresh::Doc(doc, etag) = rt.block_on(endpoint.repo_bytes(name, None))? else {
            return fail(Kind::NotFound, format!("no repo {name} at {}", endpoint.url()));
        };
        let before = first.get_or_insert_with(|| doc.clone()).clone();
        rt.block_on(old_objects(endpoint, client, scratch, &doc))?;
        let mut m = migrator.repo(&doc, &|o| Ok(scratch.read(o)?))?;
        if refused > 0 {
            m.report["refused_puts"] = json!(refused);
        }
        let bytes = m.repo.to_json()?;
        if bytes == doc {
            return Ok(Done { report: m.report, before });
        }
        let sent = rt.block_on(upload(endpoint, &m.objects))?;
        if rt.block_on(endpoint.put_repo_bytes(name, bytes, Some(&etag)))? {
            m.report["sent"] = json!({"objects": sent.0, "bytes": sent.1});
            return Ok(Done { report: m.report, before });
        }
        refused += 1;
    }
}

/// The manifests and checkpoints `doc` names, then each manifest's header and files.
async fn old_objects(endpoint: &Arc<Endpoint>, client: &Client, scratch: &Store, doc: &[u8]) -> Result<()> {
    let old: OldRepo = serde_json::from_slice(doc).map_err(|e| invalid(format!("repo document: {e}")))?;
    let lanes = old.releases.iter().flat_map(|r| r.lanes.iter().flat_map(|l| l.values()));
    let roots: Vec<ObjectRef> = lanes.chain(&old.checkpoints).cloned().collect();
    let (origin, want): (Arc<dyn Origin>, Want) = (endpoint.clone(), Want::default());
    let fetcher = Fetcher::new(Arc::new(scratch.clone()), client.clone(), None, fetch::Config::default());
    fetcher.objects(&origin, &roots, &want).await?;
    let (mut files, mut v1_headers) = (Vec::new(), Vec::new());
    for root in &roots {
        let raw = scratch.read(root)?;
        if let Some(m) = v1::manifest(&raw)? {
            files.extend(m.files.into_iter().map(|(_, o)| o));
            files.push(m.header.clone());
            v1_headers.push(m.header);
            continue;
        }
        let doc: Option<Doc> = serde_json::from_slice(&raw).ok();
        files.extend(doc.filter(|d| d.format.starts_with("tensorfs.manifest/")).into_iter().flat_map(|d| d.files.into_values()));
    }
    fetcher.objects(&origin, &files, &want).await?;
    // A v1 header's assets are objects of their own.
    let mut assets = Vec::new();
    for h in &v1_headers {
        assets.extend(v1::assets(&scratch.read(h)?)?);
    }
    fetcher.objects(&origin, &assets, &want).await?;
    Ok(())
}

/// Sends what `endpoint` lacks of `objects`: how many it sent, and their bytes.
async fn upload(endpoint: &Endpoint, objects: &Objects) -> Result<(usize, u64)> {
    let mut sent = (0, 0);
    for (object, bytes) in objects {
        let bytes = Bytes::from(bytes.clone());
        if endpoint.upload(object, &move || Ok(Payload::Bytes(bytes.clone()))).await? {
            sent = (sent.0 + 1, sent.1 + object.length);
        }
    }
    Ok(sent)
}

/// The store at `root` in place: its pins go (a store has none now), the format record becomes
/// this build's, and each repo of `refs/` migrates through the store as a directory endpoint.
pub fn store(root: &Path, migrator: &mut Migrator) -> Result<Json> {
    let root = root.canonicalize()?;
    let dropped = fs::read_dir(root.join("pins")).map_or(0, |d| d.count());
    if dropped > 0 {
        fs::remove_dir_all(root.join("pins"))?;
    }
    let io = |e: platform::Error| Failure { kind: Kind::Io, detail: e.to_string() };
    let format = format!("{{\"format\":\"tensorfs.store/{}\"}}", crate::STORE_FORMAT);
    platform::durable::replace(&root.join("format"), format.as_bytes(), 0o644).map_err(io)?;
    let store = Store::open(&root)?;
    let client = Client::new(SourcePolicy::default());
    let endpoint = Arc::new(Endpoint::new(&client, &root.to_string_lossy(), None)?);
    let mut repos = Vec::new();
    for org in fs::read_dir(root.join("refs"))? {
        let org = org?.path();
        for file in fs::read_dir(&org)? {
            let path = file?.path();
            if let (Some(o), Some(name), Some("json")) = (org.file_name(), path.file_stem(), path.extension().and_then(|e| e.to_str())) {
                let name = format!("{}/{}", o.to_string_lossy(), name.to_string_lossy());
                repos.push(migrate(&endpoint, &client, &name, &store, migrator)?.report);
            }
        }
    }
    Ok(json!({"store": root, "repos": repos, "pins_dropped": dropped}))
}
