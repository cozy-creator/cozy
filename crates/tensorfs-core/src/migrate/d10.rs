//! xs-053 A11: models from before D10 (a `tensorfs.manifest/1` per lane, its header in the
//! five-field layout) become D10: a cozytensors per lane, a tree per release. Header-only:
//! every chunk, tensor, part and location stays as it was. Run data a lane lacks comes from
//! its family's pinned upstream ([`UPSTREAMS`], owner Decision 1) and is recorded in the
//! header's `filled`. Deleted once it has run.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use minicbor::Decoder;
use serde::Deserialize;
use serde_json::{json, Value as Json};

use crate::failure::{fail, Failure, Kind, Result};
use crate::fetch::{self, Client, Endpoint, Fetcher, Fresh, Origin, Payload, SourcePolicy, Want};
use crate::format::{Component, Config, Header, ObjectRef, Release, Repo, Tree, Value, HEADER_PATH};
use crate::import::{layout, remote};
use crate::store2::{Pin, Store};

/// A family's pinned upstream and what it fills where a lane has none: a path, a folder
/// (`tokenizer/`), or `README`/`LICENSE` (any top-level file so named).
pub struct Upstream {
    /// `org/name` patterns, `*` any run of characters.
    pub models: &'static [&'static str],
    pub from: &'static str,
    pub files: &'static [&'static str],
    /// The weights' pinned source, when it is not `from`.
    pub source: Option<&'static str>,
    pub note: &'static str,
}

const NOTE: &str = "run data the lane lacked, from its family's upstream at a pinned commit";
const SDXL: &str = "hf://stabilityai/stable-diffusion-xl-base-1.0@462165984030d82259a11f4367a4eed129e94a7b";
const PIPELINE: [&str; 3] = ["model_index.json", "tokenizer/", "tokenizer_2/"];
const DOCS: [&str; 2] = ["README", "LICENSE"];

const fn row(models: &'static [&'static str], from: &'static str, files: &'static [&'static str]) -> Upstream {
    Upstream { models, from, files, source: None, note: NOTE }
}

/// The owner-approved table (2026-10-19). Civitai fine-tunes get run data only.
pub const UPSTREAMS: &[Upstream] = &[
    row(&["*/sdxl", "paul/d1-*sdxl*"], SDXL, &["model_index.json", "tokenizer/", "tokenizer_2/", "README", "LICENSE"]),
    row(
        &[
            "*/babes-pony", "*/big-love-xl", "*/cyberrealistic-pony", "*/cyberrealistic-xl",
            "*/diving-illustrious-anime", "*/furrytoonmix-xl", "*/gonzalomo-xl", "*/nova-anime-xl",
            "*/nova-furry-xl", "*/nova-orange-xl", "*/perfectdeliberate-anime", "*/pie-models-mango",
            "*/prefect-illustrious-xl", "*/realism-illustrious", "*/wai-illustrious", "*/civitai-2831949",
        ],
        SDXL,
        &PIPELINE,
    ),
    row(&["*/anima"], ANIMA, &["model_index.json", "tokenizer/", "README", "LICENSE"]),
    row(&["*/anima-civitai"], ANIMA, &["model_index.json", "tokenizer/"]),
    row(&["*/flux-3-klein"], "hf://black-forest-labs/flux-3-image-klein@8454017f9ebaada30c998fccd9f67a80c7580f68", &KLEIN),
    row(&["*/flux-3-klein-base"], "hf://black-forest-labs/flux-3-image-klein-base@1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4", &KLEIN),
    row(
        &["*/flux-3-klein-upsampler"],
        "hf://black-forest-labs/flux-3-image-klein-pup@8b069751eb5dcab23829d9c5eccb470eb8b8a4ed",
        &["special_tokens_map.json", "README", "LICENSE"],
    ),
    row(
        &["*/qwen-image-2.1", "*/reference-image"],
        "hf://Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758",
        &["model_index.json", "scheduler/", "tokenizer/", "README", "LICENSE"],
    ),
    row(
        &["*/qwen3.5-9b", "*/qwen3.5-9b-mtp"],
        "hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        &["generation_config.json", "special_tokens_map.json", "README", "LICENSE"],
    ),
    row(
        &["*/minimax-h3", "*/minimax-h3-vsa-gate", "*/minimax-h3-turbo-exp", "*/minimax-h3-turbo-lora"],
        "hf://MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe",
        &DOCS,
    ),
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
        files: &DOCS,
        source: Some("hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8"),
        note: NOTE,
    },
    row(&["*/birefnet"], "hf://ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", &["README"]),
];
const ANIMA: &str = "hf://circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9db359c31ad0e8aa268d15775473c2176c";
const KLEIN: [&str; 3] = ["model_index.json", "scheduler/", "tokenizer/"];

/// Old metadata D10's loading metadata drops; `sources` becomes `source`.
const DROPPED: [&str; 4] = ["normalized", "converters", "sources", "migrated_from"];

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

/// Whether `path` is under `entry`: a folder's files, or a top-level file named by it.
fn under(entry: &str, path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    match entry.ends_with('/') {
        true => path.starts_with(&entry.to_ascii_lowercase()),
        false => !path.contains('/') && path.starts_with(&entry.to_ascii_lowercase()),
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
    #[serde(default)]
    revision: u64,
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
    /// they differ) at its next revision, yanked or not; each other old checkpoint a tree of
    /// its own (`model.cozytensors`).
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
                    releases.push(Release { version: r.version.clone(), tree: tree.clone(), yanked: r.yanked, revision: r.revision });
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
            let release = Release { version: r.version.clone(), tree: put(&mut objects, &tree)?, yanked: r.yanked, revision: r.revision + 1 };
            reports.push(json!({"version": release.version, "tree": release.tree.id(), "yanked": release.yanked,
                "revision": release.revision, "lanes": report, "conflicts": conflicts}));
            releases.push(release);
        }
        let cited: Vec<&ObjectRef> = old.releases.iter().flat_map(|r| r.lanes.iter().flat_map(|l| l.values())).collect();
        let (mut checkpoints, mut unreleased) = (Vec::new(), Vec::new());
        for c in old.checkpoints.iter().filter(|c| !cited.contains(c)) {
            let doc: Doc = serde_json::from_slice(&read(c)?).map_err(|e| invalid(format!("{}: {e}", c.id())))?;
            if doc.format.starts_with("tensorfs.tree/") {
                checkpoints.push(c.clone());
                continue;
            }
            let lane = self.lane(c, upstream, read, &mut objects)?;
            let mut tree = Tree { files: lane.files.clone() };
            tree.set_variant("", lane.header.clone());
            let at = put(&mut objects, &tree)?;
            unreleased.push(json!({"old": c.id(), "tree": at.id(), "lane": lane.report}));
            checkpoints.push(at);
        }
        let repo = Repo { name: old.name.clone(), releases, checkpoints };
        let report = json!({"model": old.name, "upstream": upstream.map(|u| u.from), "releases": reports, "checkpoints": unreleased});
        Ok(Migrated { repo, objects, report })
    }

    /// One old lane: its header in D10 (old tensors, filled run data, configs, assets) and its
    /// repo files; every lane file is checked to have landed in one of them.
    fn lane(&mut self, manifest: &ObjectRef, upstream: Option<&Upstream>, read: Read, objects: &mut Objects) -> Result<Lane> {
        let doc: Doc = serde_json::from_slice(&read(manifest)?).map_err(|e| invalid(format!("{}: {e}", manifest.id())))?;
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
        let mut filled = Vec::new();
        for (u, entry) in upstream.iter().flat_map(|u| u.files.iter().map(move |e| (u, e))) {
            for (path, b) in self.upstream(u.from, entry)? {
                if !bytes.contains_key(&path) {
                    filled.push(path.clone());
                    bytes.insert(path, b);
                }
            }
        }
        let tensored: Vec<String> = old.components().iter().map(|c| c.name.clone()).collect();
        let l = layout(bytes.clone(), &tensored)?;
        let mut keep = |b: Vec<u8>| add(objects, b);
        let components = (l.components.into_iter())
            .map(|f| Component {
                tensors: old.component(&f.name).map(|c| c.tensors.clone()).unwrap_or_default(),
                assets: f.assets.into_iter().map(|(p, b)| (p, keep(b))).collect(),
                name: f.name,
                class: f.class,
                config: f.config,
                config_file: f.config_file,
            })
            .collect();
        let repo_files: BTreeMap<String, ObjectRef> = l.files.into_iter().map(|(p, b)| (p, keep(b))).collect();
        let mut metadata = old.metadata.clone();
        let sources = metadata.remove("sources");
        DROPPED.iter().for_each(|k| drop(metadata.remove(*k)));
        let source = match (upstream, sources) {
            (Some(u), _) => Some(Value::Str(u.source.unwrap_or(u.from).to_owned())),
            (None, Some(Value::List(mut s))) if s.len() == 1 => Some(s.remove(0)),
            (None, s) => s,
        };
        metadata.extend(source.map(|s| ("source".to_owned(), s)));
        if let (Some(u), false) = (upstream, filled.is_empty()) {
            let text = |s: &str| Value::Str(s.to_owned());
            metadata.insert("filled".into(), Value::Map(BTreeMap::from([
                ("from".to_owned(), text(u.from)),
                ("files".to_owned(), Value::List(filled.iter().map(|f| text(f)).collect())),
                ("note".to_owned(), text(u.note)),
            ])));
        }
        let new = Header::new(old.chunks.clone(), l.pipeline, components, metadata)?;
        let cbor = new.to_cbor()?;
        check(&old, &new, &cbor, &bytes, &repo_files)?;
        let at = add(objects, cbor);
        let mut report = json!({"old": manifest.id(), "header": at.id()});
        if !filled.is_empty() {
            report["filled"] = json!(filled);
        }
        Ok(Lane { header: at, files: repo_files, report })
    }

    /// The files under `entry` at the pinned upstream `from`, by path.
    fn upstream(&mut self, from: &str, entry: &str) -> Result<Vec<(String, Vec<u8>)>> {
        if !self.listings.contains_key(from) {
            let listing = remote::list(&remote::parse(from)?, &self.access)?;
            self.listings.insert(from.to_owned(), listing.members);
        }
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

/// The D10 header against the old one and the lane's files: the same chunks and tensors, its
/// bytes reading back to it, and each file a config, an asset or a repo file.
fn check(old: &Header, new: &Header, cbor: &[u8], files: &BTreeMap<String, Vec<u8>>, repo: &BTreeMap<String, ObjectRef>) -> Result<()> {
    let integrity = |what: String| fail(Kind::Integrity, what);
    if new.chunks != old.chunks || Header::from_cbor(cbor)? != *new {
        return integrity("the chunks moved, or the header does not read back".into());
    }
    if let Some((c, t)) = old.tensors().find(|(c, t)| new.tensor(c, &t.key) != Some(t)) {
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

/// Repo `name` at `endpoint` in D10: the old objects it cites fetched into `scratch` over
/// `client`, what the D10 document adds uploaded where missing, then the document replaced while
/// it is as read (read and migrated again when another writer won).
pub fn migrate(endpoint: &Arc<Endpoint>, client: &Client, name: &str, scratch: &Store, migrator: &mut Migrator) -> Result<Json> {
    let rt = fetch::runtime()?;
    loop {
        let Fresh::Doc(doc, etag) = rt.block_on(endpoint.repo_bytes(name, None))? else {
            return fail(Kind::NotFound, format!("no repo {name} at {}", endpoint.url()));
        };
        rt.block_on(old_objects(endpoint, client, scratch, &doc))?;
        let mut m = migrator.repo(&doc, &|o| Ok(scratch.read(o)?))?;
        let bytes = m.repo.to_json()?;
        if bytes == doc {
            return Ok(m.report);
        }
        let sent = rt.block_on(upload(endpoint, &m.objects))?;
        if rt.block_on(endpoint.put_repo_bytes(name, bytes, Some(&etag)))? {
            m.report["sent"] = json!({"objects": sent.0, "bytes": sent.1});
            return Ok(m.report);
        }
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
    let mut files = Vec::new();
    for root in &roots {
        let doc: Option<Doc> = serde_json::from_slice(&scratch.read(root)?).ok();
        files.extend(doc.filter(|d| d.format.starts_with("tensorfs.manifest/")).into_iter().flat_map(|d| d.files.into_values()));
    }
    fetcher.objects(&origin, &files, &want).await?;
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

/// The store at `root` in place: pins naming old objects go, the format record becomes this
/// build's, and each repo of `refs/` migrates through the store as a directory endpoint.
pub fn store(root: &Path, migrator: &mut Migrator) -> Result<Json> {
    let root = root.canonicalize()?;
    let mut dropped = 0;
    for file in fs::read_dir(root.join("pins")).into_iter().flatten() {
        let path = file?.path();
        let pin: Option<Pin> = fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let old = |o: &ObjectRef| {
            let doc = |b: &[u8]| serde_json::from_slice::<Doc>(b).is_ok_and(|d| d.format.starts_with("tensorfs.manifest/"));
            fs::read(root.join(o.key())).is_ok_and(|b| b.first() == Some(&0x85) || doc(&b))
        };
        if pin.is_none_or(|p| p.roots.iter().any(old)) {
            fs::remove_file(&path)?;
            dropped += 1;
        }
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
                repos.push(migrate(&endpoint, &client, &name, &store, migrator)?);
            }
        }
    }
    Ok(json!({"store": root, "repos": repos, "pins_dropped": dropped}))
}
