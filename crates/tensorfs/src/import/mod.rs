//! Convert: sources become one canonical model in the D10 layout (normalize at ingest).
//!
//! 1. List and pin each source, and select what the model is built from (`remote`).
//! 2. Download remote members into `<tmp>/import/<id>/`, never the store; resumable, since
//!    `<id>` is the pinned selection's digest, and locked (`<id>.lock`) while one convert uses
//!    it. Local members are read where they are.
//! 3. Fold quantized carriers into encoded tensors, by their producer's declaration
//!    (`carriers`). Recognize from safetensors headers alone: a built-in table claims a group
//!    when its key set is the table's (`engine`). Unclaimed groups are stored as they are.
//! 4. Map every other file ([`layout`]): `model_index.json` becomes the pipeline, a component's
//!    config its structured config, the other run files in its folder its assets; README,
//!    LICENSE, images, code and docs stay repo files. A table's pinned reference supplies the
//!    configs and tokenizers the source lacks (recorded as `filled`).
//! 5. Convert row by row into the [`Sink`], then delete the tmp area.
//!
//! A GGUF file is one component of its own (`gguf`).

pub mod carriers;
pub mod engine;
pub mod gguf;
pub mod lora;
pub mod order;
pub mod remote;
pub mod source;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{
    ClassRef, Config, ConfigValue, Dtype, Encoding, Metadata, ObjectRef, Pipeline, Value,
    CONFIG_FILE,
};
use crate::store::Error;
use crate::writer::{Data, PartData, Source, Writer};

pub use engine::{tables, Table};
pub use order::{constructor_order, Ordered};
pub use remote::Access;

/// One stored part of an output tensor, `data` yielding exactly its bytes.
pub struct Part<'a> {
    /// `""` for a plain tensor's one part.
    pub role: &'a str,
    pub dtype: &'a Dtype,
    pub shape: &'a [u64],
    pub data: &'a mut dyn Read,
}

/// Where a convert's output goes: the writer, or anything that takes a D10 model.
pub trait Sink {
    /// The variant ([`main_variant`]), the pipeline, each component's class, config and assets,
    /// and the repo files; before any tensor.
    fn begin(&mut self, variant: &str, layout: &Layout) -> Result<()>;
    /// One tensor: plain (no encoding, one part) or encoded (its parts in order).
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        encoding: Option<&Encoding>,
        parts: Vec<Part>,
    ) -> Result<()>;
    /// Commits with the header's loading metadata ([`Imported::metadata`]).
    fn commit(&mut self, metadata: Metadata) -> Result<()>;
}

/// A convert writes through the writer; tensors an interrupted convert journaled are skipped.
impl Sink for Writer {
    fn begin(&mut self, _: &str, layout: &Layout) -> Result<()> {
        if let Some(p) = &layout.pipeline {
            self.pipeline(p.class.clone(), p.config.clone());
        }
        for c in &layout.components {
            let config_file = c.config_file.as_deref();
            let mut w = (self.component(&c.name, c.class.clone(), c.config.clone(), config_file))
                .map_err(Error::refusal)?;
            for (name, bytes) in &c.assets {
                w.asset(name, bytes).map_err(Error::refusal)?;
            }
        }
        for (path, bytes) in &layout.files {
            self.repo_file(path, bytes).map_err(Error::refusal)?;
        }
        Ok(())
    }

    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &Dtype,
        shape: &[u64],
        encoding: Option<&Encoding>,
        parts: Vec<Part>,
    ) -> Result<()> {
        if self.has(component, key) {
            return Ok(());
        }
        let mut parts = parts.into_iter().map(|p| PartData {
            role: p.role.to_owned(),
            dtype: p.dtype.clone(),
            shape: p.shape.to_vec(),
            data: Source::Read(p.data),
        });
        let data = match encoding {
            Some(encoding) => Data::Encoded {
                encoding: encoding.clone(),
                parts: parts.collect(),
            },
            None => Data::Plain(parts.next().expect("a plain tensor's part").data),
        };
        (self.get(component).put(key, dtype, shape, data)).map_err(Error::refusal)
    }

    fn commit(&mut self, metadata: Metadata) -> Result<()> {
        for (key, value) in metadata {
            self.set_metadata(&key, value);
        }
        Writer::commit(self).map(drop).map_err(Error::refusal)
    }
}

/// Converts `req` into the plain folder `dir`: its repo files, `model.<variant>.cozytensors`
/// (the variant [`main_variant`]) and `chunks/`. An interrupted convert into `dir` resumes. The
/// header.
pub fn convert_folder(req: &Request, dir: &Path) -> Result<ObjectRef> {
    struct Folder<'d>(&'d Path, Option<Writer>);
    impl Folder<'_> {
        fn writer(&mut self) -> &mut Writer {
            self.1.as_mut().expect("begun")
        }
    }
    impl Sink for Folder<'_> {
        fn begin(&mut self, variant: &str, layout: &Layout) -> Result<()> {
            let writer = Writer::folder(self.0, variant, None, true).map_err(Error::refusal)?;
            self.1.insert(writer).begin(variant, layout)
        }
        fn put(
            &mut self,
            component: &str,
            key: &str,
            dtype: &Dtype,
            shape: &[u64],
            encoding: Option<&Encoding>,
            parts: Vec<Part>,
        ) -> Result<()> {
            self.writer()
                .put(component, key, dtype, shape, encoding, parts)
        }
        fn commit(&mut self, metadata: Metadata) -> Result<()> {
            Sink::commit(self.writer(), metadata)
        }
    }
    let mut folder = Folder(dir, None);
    convert(req, &mut folder)?;
    Ok(folder
        .writer()
        .committed()
        .expect("a convert commits")
        .clone())
}

/// What a component's folder held besides weights.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Folder {
    pub name: String,
    pub class: ClassRef,
    pub config: Config,
    /// The config's file name when it isn't `config.json`.
    pub config_file: Option<String>,
    /// Run files by path relative to the folder, byte-exact.
    pub assets: BTreeMap<String, Vec<u8>>,
}

/// A source's non-weight files in the D10 layout.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layout {
    pub pipeline: Option<Pipeline>,
    /// Header order: components with tensors in construction order, then the rest by name.
    pub components: Vec<Folder>,
    /// Repo files by path, byte-exact; never inside the cozytensors.
    pub files: BTreeMap<String, Vec<u8>>,
}

/// What `model_index.json` holds: the pipeline.
pub const PIPELINE_FILE: &str = "model_index.json";
/// The component of a model without a pipeline, whose folder is the root.
pub const ROOT_COMPONENT: &str = "model";
/// A folder's config file: the first of these it holds.
const CONFIGS: [&str; 5] = [
    CONFIG_FILE,
    "scheduler_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
];

/// Header metadata: the pinned source (a list when there are several).
pub const SOURCE: &str = "source";
/// Header metadata: run data a pinned reference supplied, `{from, files, note}`.
pub const FILLED: &str = "filled";
/// The header metadata key of each component's source safetensors `__metadata__`
/// (`{component: {key: value}}`), which export writes back.
pub const FILE_METADATA: &str = "safetensors_metadata";

/// `__metadata__` keys that declare a quantized carrier, which the encodings now state.
const DECLARATIONS: [&str; 2] = ["_quantization_metadata", "quantization_config"];

/// What a convert reports; the header's metadata comes from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imported {
    /// Every key is canonical: converted by a table, or the source's own Diffusers or
    /// Transformers layout. `false` means stored as-is from a layout nothing recognized.
    pub normalized: bool,
    pub converters: Vec<String>,
    /// The pinned sources.
    pub sources: Vec<String>,
    /// Each pinned reference that supplied run data the source lacks, with the paths.
    pub filled: Vec<(String, Vec<String>)>,
    /// Each component's source safetensors `__metadata__` (LoRA training fields, `format`, …).
    pub file_metadata: BTreeMap<String, BTreeMap<String, String>>,
}

impl Imported {
    /// The header's loading metadata: `source`, `filled` and each component's `__metadata__`.
    pub fn metadata(&self) -> Metadata {
        let text = |s: &str| Value::Str(s.to_owned());
        let texts = |l: &[String]| Value::List(l.iter().map(|s| text(s)).collect());
        let one = |mut l: Vec<Value>| match l.len() {
            1 => l.remove(0),
            _ => Value::List(l),
        };
        let mut out = Metadata::new();
        out.insert(
            SOURCE.into(),
            one(self.sources.iter().map(|s| text(s)).collect()),
        );
        let filled = (self.filled.iter()).map(|(from, files)| {
            Value::Map(BTreeMap::from([
                ("from".to_owned(), text(from)),
                ("files".to_owned(), texts(files)),
                (
                    "note".to_owned(),
                    text("run data the source lacks, from a pinned reference"),
                ),
            ]))
        });
        if !self.filled.is_empty() {
            out.insert(FILLED.into(), one(filled.collect()));
        }
        if !self.file_metadata.is_empty() {
            let strings = |m: &BTreeMap<String, String>| {
                Value::Map(m.iter().map(|(k, v)| (k.clone(), text(v))).collect())
            };
            let by_component = (self.file_metadata.iter()).map(|(c, m)| (c.clone(), strings(m)));
            out.insert(FILE_METADATA.into(), Value::Map(by_component.collect()));
        }
        out
    }
}

pub struct Request<'a> {
    pub sources: &'a [String],
    /// A table to convert with; required for tables recognition never picks alone.
    pub converter: Option<&'a str>,
    /// The source's Diffusers weights variant (`fp16`), when a folder has several.
    pub source_variant: Option<&'a str>,
    pub access: &'a Access,
    /// The store's tmp directory.
    pub tmp: &'a Path,
    pub progress: &'a (dyn Fn(&str, u64, u64) + Sync),
    /// Asked before members/tensors and while network work is pending. True stops before
    /// finalization begins; an already-started commit cannot be undone.
    pub stop: Option<&'a (dyn Fn() -> bool + Sync)>,
}

fn stopped(req: &Request) -> Result<()> {
    remote::Scope::new(req.access, req.stop).check()
}

fn digest(parts: &[&str]) -> String {
    crate::sha256::hex_digest(parts.join("\n").as_bytes())
}

/// Lands `members` under `area`: `(path, local file)` pairs.
fn land(
    members: &[remote::Member],
    area: &Path,
    req: &Request<'_>,
) -> Result<Vec<(String, PathBuf)>> {
    let access = remote::Scope::new(req.access, req.stop);
    let progress = req.progress;
    let total = members.iter().filter_map(|m| m.length).sum();
    let mut done = 0;
    let mut out = Vec::new();
    for m in members {
        access.check()?;
        progress("downloading", done, total);
        access.check()?;
        out.push((m.path.clone(), access.land(m, &area.join(&m.path))?));
        done += m.length.unwrap_or(0);
    }
    progress("downloading", done, total);
    access.check()?;
    Ok(out)
}

fn folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(f, _)| f)
}

fn name(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(_, n)| n)
}

fn join(folder: &str, path: &str) -> String {
    match folder {
        "" => path.to_owned(),
        f => format!("{f}/{path}"),
    }
}

fn is_gguf(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

/// Library versions by library, as configs record them.
pub type Versions = BTreeMap<String, String>;

/// Lands `docs` (a pinned reference's) and adds each at `into/<path>` where `files` has none;
/// the paths added. Every library version a landed config records joins `versions`.
fn fill(
    docs: &[remote::Member],
    into: &str,
    area: &Path,
    req: &Request,
    files: &mut BTreeMap<String, Vec<u8>>,
    versions: &mut Versions,
) -> Result<Vec<String>> {
    let mut added = Vec::new();
    for (path, local) in land(docs, area, req)? {
        let path = join(into, &path);
        let bytes = std::fs::read(&local)
            .map_err(|e| crate::err::classify_io(format!("read {path}"), e))?;
        if let Ok(config) = serde_json::from_slice::<Config>(&bytes) {
            for (library, version) in recorded(&config) {
                versions.entry(library).or_insert(version);
            }
        }
        if !files.contains_key(&path) {
            files.insert(path.clone(), bytes);
            added.push(path);
        }
    }
    Ok(added)
}

/// The configs, tokenizers and license at the root of the repository a GGUF names as its
/// source, added in the GGUF's folder `into` where the source has none: the pinned reference
/// and the paths added.
fn gguf_documents(
    reference: &str,
    into: &str,
    req: &Request,
    area: &Path,
    files: &mut BTreeMap<String, Vec<u8>>,
    versions: &mut Versions,
) -> Result<(String, Vec<String>)> {
    let listing = remote::Scope::new(req.access, req.stop).list(&remote::parse(reference)?)?;
    let pin = listing.pinned.clone();
    let read = |m: &remote::Member| remote::Scope::new(req.access, req.stop).read_small(m);
    let docs: Vec<remote::Member> = remote::select(listing, None, remote::Files::Configs, &read)?
        .into_iter()
        .filter(|m| !remote::weights(&m.path) && folder(&m.path).is_empty())
        .collect();
    let added = fill(&docs, into, &area.join("gguf-reference"), req, files, versions)?;
    Ok((pin, added))
}

/// The model's files: every non-weight file of the source, byte-exact at its path, except a
/// converted folder's `config.json` (the foreign layout's); each table's reference (its index,
/// the produced components' configs and the asset folders) where the source has none; the
/// tables' inline files. With each reference that filled some, and the paths, and the library
/// versions the references record.
type Documents = (BTreeMap<String, Vec<u8>>, Vec<(String, Vec<String>)>, Versions);

fn documents(
    set: &source::SourceSet,
    recognized: &engine::Recognized,
    planned: &[engine::Planned],
    req: &Request,
    area: &Path,
) -> Result<Documents> {
    let tables = tables();
    let converted: Vec<&str> = recognized
        .claims
        .iter()
        .map(|c| set.groups[c.group].folder.as_str())
        .collect();
    let raw: Vec<usize> = recognized
        .tables
        .iter()
        .flat_map(|t| &tables[*t].components)
        .flat_map(|c| &c.rows)
        .filter_map(|r| match &r.source {
            engine::Src::Raw(raw) => set.find_raw(&raw.sha256, raw.length).map(|(f, _)| f),
            engine::Src::Key(_) => None,
        })
        .collect();
    let mut files = BTreeMap::new();
    for p in &set.plain {
        let foreign = converted.contains(&folder(&p.path)) && name(&p.path) == CONFIG_FILE;
        if !raw.contains(&p.file) && !foreign && !is_gguf(&p.path) {
            files.insert(p.path.clone(), set.read_plain(p)?);
        }
    }
    let produced: Vec<&str> = planned.iter().map(|p| p.component.as_str()).collect();
    let (mut filled, mut versions) = (Vec::new(), Versions::new());
    for t in &recognized.tables {
        let table = &tables[*t];
        if let Some(reference) = &table.reference {
            let listing =
                remote::Scope::new(req.access, req.stop).list(&remote::parse(reference)?)?;
            let pin = listing.pinned.clone();
            let read = |m: &remote::Member| remote::Scope::new(req.access, req.stop).read_small(m);
            let selected = remote::select(listing, None, remote::Files::Configs, &read)?;
            let weighted: Vec<&str> = selected
                .iter()
                .filter(|m| remote::weights(&m.path))
                .map(|m| folder(&m.path))
                .collect();
            let docs: Vec<remote::Member> = selected
                .iter()
                .filter(|m| {
                    let f = folder(&m.path);
                    !remote::weights(&m.path)
                        && (f.is_empty() || produced.contains(&f) || !weighted.contains(&f))
                })
                .cloned()
                .collect();
            let added = fill(&docs, "", &area.join("reference"), req, &mut files, &mut versions)?;
            if !added.is_empty() {
                filled.push((pin, added));
            }
        }
        for (path, doc) in &table.files {
            files.insert(path.clone(), serde_json::to_vec(doc).expect("a JSON value"));
        }
    }
    Ok((files, filled, versions))
}

/// Maps a source's non-weight `files` to D10 for the components `tensored` (in construction
/// order):
/// - `model_index.json` is the pipeline: class `diffusers` `_class_name`, config the parsed JSON.
///   Each entry `name: [library, Class]` is a component in folder `name`, tensors or not
///   (schedulers, tokenizers, processors), of that library and class as written; a component
///   it names no class for has the unknown class.
/// - Without one, a component named [`ROOT_COMPONENT`] owns the root (a Transformers folder),
///   and a component's class is its config's ([`config_class`]).
/// - A pair's library that is a Diffusers pipeline module (`stable_diffusion`) is `diffusers`
///   ([`class_of`]). A class's version is its own config's for its library (`_diffusers_version`,
///   `transformers_version`), else one another config of the model records for that library,
///   else `reference`'s (the converter's pinned reference configs), else `""`.
/// - In a component's folder, the first of [`CONFIGS`] is its config, and the other run files
///   ([`run_data`]) its assets.
/// - Every other file is a repo file. A component with no tensors, config or assets is none.
pub fn layout(
    mut files: BTreeMap<String, Vec<u8>>,
    tensored: &[String],
    reference: &Versions,
) -> Result<Layout> {
    let mut pipeline = match files.remove(PIPELINE_FILE) {
        Some(bytes) => {
            let config = parse_config(PIPELINE_FILE, &bytes)?;
            let class = (config.get("_class_name").and_then(ConfigValue::as_str))
                .map_or_else(ClassRef::default, |c| ClassRef::new("diffusers", "", c));
            Some(Pipeline { class, config })
        }
        None => None,
    };
    let listed: BTreeMap<String, ClassRef> = (pipeline.iter())
        .flat_map(|p| &p.config)
        .filter(|(name, _)| !name.starts_with('_'))
        .filter_map(|(name, row)| match row.as_array()?.as_slice() {
            [library, class, ..] => Some((
                name.clone(),
                ClassRef::new(library.as_str()?, "", class.as_str()?),
            )),
            _ => None,
        })
        .collect();
    let mut names = tensored.to_vec();
    names.extend(listed.keys().filter(|n| !tensored.contains(n)).cloned());
    let root = pipeline.is_none()
        && (names.is_empty() || names.iter().any(|n| n == ROOT_COMPONENT))
        && !files.keys().any(|p| p.starts_with("model/"));
    if root && names.is_empty() {
        names.push(ROOT_COMPONENT.into());
    }
    let mut components = Vec::new();
    for name in names {
        let at = if root && name == ROOT_COMPONENT {
            ""
        } else {
            name.as_str()
        };
        let inside = |p: &str| match at {
            "" => Some(p.to_owned()).filter(|p| !p.contains('/')),
            f => (p.strip_prefix(f)?.strip_prefix('/')).map(str::to_owned),
        };
        let mine: Vec<(String, String)> = (files.keys())
            .filter_map(|p| Some((p.clone(), inside(p)?)))
            .filter(|(_, rel)| run_data(rel))
            .collect();
        let mut assets: BTreeMap<String, Vec<u8>> = (mine.into_iter())
            .map(|(path, rel)| (rel, files.remove(&path).expect("listed")))
            .collect();
        let file = CONFIGS.iter().find(|c| assets.contains_key(**c));
        let config = match file {
            Some(f) => parse_config(&join(at, f), &assets.remove(*f).expect("found"))?,
            None => Config::new(),
        };
        if file.is_none() && assets.is_empty() && !tensored.contains(&name) {
            continue;
        }
        let class = match (&pipeline, file) {
            (Some(_), _) => listed.get(&name).cloned().unwrap_or_default(),
            (None, Some(f)) => config_class(f, &config),
            (None, None) => ClassRef::default(),
        };
        components.push(Folder {
            class,
            name,
            config,
            config_file: file.filter(|f| **f != CONFIG_FILE).map(|f| f.to_string()),
            assets,
        });
    }
    let mut versions = Versions::new();
    let configs = pipeline.iter().map(|p| &p.config).chain(components.iter().map(|c| &c.config));
    for (library, version) in configs.flat_map(recorded).chain(reference.clone()) {
        versions.entry(library).or_insert(version);
    }
    let classes = (pipeline.iter_mut().map(|p| (&mut p.class, &p.config)))
        .chain(components.iter_mut().map(|c| (&mut c.class, &c.config)));
    for (class, config) in classes.filter(|(class, _)| !class.is_unknown()) {
        let model = class_of(&class.library, &class.name, &versions);
        let mut own = recorded(config).filter(|(library, _)| *library == model.library);
        let version = own.next().map_or(model.version, |(_, version)| version);
        *class = ClassRef { version, ..model };
    }
    Ok(Layout {
        pipeline,
        components,
        files,
    })
}

/// A class from its library and name: the library as written when it is installed (`diffusers`,
/// `transformers`, or one `versions` holds), else a Diffusers pipeline module (`stable_diffusion`,
/// `kandinsky`) and so `diffusers`; at the version `versions` holds for that library, else `""`.
/// `versions` is the model's: its configs' records, then the converter reference's.
pub fn class_of(library: &str, name: &str, versions: &Versions) -> ClassRef {
    let library = match library {
        "diffusers" | "transformers" => library,
        installed if versions.contains_key(installed) => installed,
        _ => "diffusers",
    };
    ClassRef::new(library, versions.get(library).map_or("", String::as_str), name)
}

/// The library versions a config records: `_<library>_version` or `<library>_version` keys
/// (`_diffusers_version`, `transformers_version`).
fn recorded(config: &Config) -> impl Iterator<Item = (String, String)> + '_ {
    config.iter().filter_map(|(key, value)| {
        let library = key.strip_suffix("_version")?;
        let library = library.strip_prefix('_').unwrap_or(library);
        Some((library.to_owned(), value.as_str()?.to_owned())).filter(|(l, v)| !l.is_empty() && !v.is_empty())
    })
}

/// The variant a model is written as: its main dtype or scheme by stored bytes (`bf16`,
/// `fp8.rowwise`), `files` when it has no tensors.
pub fn main_variant<'a>(
    tensors: impl Iterator<Item = (&'a Dtype, Option<&'a Encoding>, u64)>,
) -> String {
    let mut bytes: BTreeMap<String, u64> = BTreeMap::new();
    for (dtype, encoding, n) in tensors {
        let name = encoding.map_or_else(|| dtype.name().to_ascii_lowercase(), |e| e.scheme.clone());
        *bytes.entry(name).or_default() += n;
    }
    (bytes.into_iter().max_by_key(|(_, n)| *n)).map_or_else(|| "files".to_owned(), |(name, _)| name)
}

/// Run data: a file a component loads or runs with (tokenizers and their vocabularies,
/// generation and preprocessor configs, chat templates). Docs, code and images are not.
fn run_data(path: &str) -> bool {
    let lower = name(path).to_ascii_lowercase();
    let doc = ["readme", "license", "licence", "notice", "copying"];
    !doc.iter().any(|d| lower.starts_with(d))
        && [".json", ".txt", ".model", ".jinja", ".tiktoken"]
            .iter()
            .any(|e| lower.ends_with(e))
}

/// A config file as structured JSON: floats and `null` kept.
fn parse_config(path: &str, bytes: &[u8]) -> Result<Config> {
    serde_json::from_slice(bytes).map_err(|e| Refusal {
        code: Code::MALFORMED_JSON,
        detail: format!("{path}: {e}"),
    })
}

/// A component's class without a pipeline, from its config file: Transformers'
/// `architectures[0]`, Diffusers' `_class_name`, a tokenizer config's `tokenizer_class`; the
/// unknown class when it names none. [`layout`] adds the version.
fn config_class(file: &str, config: &Config) -> ClassRef {
    let text = |key: &str| config.get(key).and_then(ConfigValue::as_str);
    let first = (config.get("architectures").and_then(ConfigValue::as_array))
        .and_then(|a| a.first()?.as_str());
    match (first, text("_class_name"), text("tokenizer_class")) {
        (Some(c), _, _) => ClassRef::new("transformers", "", c),
        (_, Some(c), _) => ClassRef::new("diffusers", "", c),
        (_, _, Some(c)) if file == "tokenizer_config.json" => ClassRef::new("transformers", "", c),
        _ => ClassRef::default(),
    }
}

pub fn convert(req: &Request, sink: &mut dyn Sink) -> Result<Imported> {
    let progress = req.progress;
    stopped(req)?;
    progress("resolving", 0, 0);
    stopped(req)?;
    let mut pinned = Vec::new();
    let mut members: Vec<remote::Member> = Vec::new();
    for raw in req.sources {
        let listing = remote::Scope::new(req.access, req.stop).list(&remote::parse(raw)?)?;
        pinned.push(listing.pinned.clone());
        let read = |m: &remote::Member| remote::Scope::new(req.access, req.stop).read_small(m);
        for m in remote::select(listing, req.source_variant, remote::Files::All, &read)? {
            if members.iter().any(|o| o.path == m.path) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("two sources both hold {}", m.path),
                );
            }
            members.push(m);
        }
    }
    let id = digest(&[
        &pinned.join("\n"),
        req.source_variant.unwrap_or(""),
        req.converter.unwrap_or(""),
    ]);
    let area = req.tmp.join("import").join(&id[..32]);
    std::fs::create_dir_all(req.tmp.join("import"))
        .map_err(|e| crate::err::classify_io("create the import area", e))?;
    // One convert of a selection at a time: the kernel's lock, released when this ends.
    let _area = platform::lock::LockFile::new(area.with_extension("lock"))
        .lock(platform::lock::Kind::Exclusive)
        .map_err(|e| {
            crate::err::classify_io("lock the import area", std::io::Error::other(e.to_string()))
        })?;
    stopped(req)?;
    let landed = land(&members, &area.join("source"), req)?;
    stopped(req)?;
    let mut set = source::SourceSet::scan(&landed)?;
    carriers::fold(&mut set)?;

    let tables = tables();
    let recognized = engine::recognize(&set, tables, req.converter)?;
    let converters: Vec<String> = recognized
        .tables
        .iter()
        .map(|t| tables[*t].name.clone())
        .collect();
    // The source's own Diffusers or Transformers layout is canonical by definition.
    let canonical = set.plain.iter().any(|p| {
        let (folder, name) = p.path.rsplit_once('/').unwrap_or(("", &p.path));
        name.ends_with(PIPELINE_FILE)
            || (name == CONFIG_FILE && set.groups.iter().any(|g| g.folder == folder))
    });
    let normalized = recognized.unclaimed.is_empty() || canonical;

    let ggufs: Vec<&source::Plain> = set.plain.iter().filter(|p| is_gguf(&p.path)).collect();
    let component = |f: &str, stem: &str| {
        let groups = set.groups.iter().filter(|o| o.folder == f).count();
        let shared = groups + ggufs.iter().filter(|p| folder(&p.path) == f).count() > 1;
        match (f, shared) {
            ("", false) => ROOT_COMPONENT.to_string(),
            ("", true) => stem.to_owned(),
            (f, false) => f.to_string(),
            (f, true) => format!("{f}/{stem}"),
        }
    };
    let identity = |g: &source::Group| component(&g.folder, &g.stem);
    let mut planned = engine::plan(&set, tables, &recognized, &identity)?;
    let (mut files, mut filled, mut versions) = documents(&set, &recognized, &planned, req, &area)?;
    let (mut converters, mut normalized) = (converters, normalized);
    let sources: BTreeMap<String, &source::Group> = (recognized.claims.iter())
        .map(|c| (c.name.clone(), &set.groups[c.group]))
        .chain((recognized.unclaimed.iter()).map(|g| (identity(&set.groups[*g]), &set.groups[*g])))
        .collect();
    let mut adapters = lora::normalize(&mut planned, &sources, &mut files)?;
    // A PEFT adapter's keys are its base model's module names: canonical once unwrapped.
    let peft = |g: &usize| adapters.get(&identity(&set.groups[*g])).is_some_and(|a| a.stated);
    if recognized.unclaimed.iter().any(peft) {
        converters.push("peft.lora".into());
        normalized |= recognized.unclaimed.iter().all(peft);
    }
    let mut ggufs_planned = Vec::new();
    for p in &ggufs {
        let file = name(&p.path);
        let component = component(folder(&p.path), &file[..file.len() - ".gguf".len()]);
        let local = &set.files[p.file];
        let opened = File::open(local)
            .map_err(|e| crate::err::classify_io(format!("open {}", local.display()), e))?;
        let plan = gguf::plan(&gguf::read(&opened)?, &component)?;
        normalized &= plan.normalized;
        converters.extend(plan.arch.as_ref().map(|a| format!("gguf.{a}")));
        let record = serde_json::to_vec(&plan.meta).expect("the record serializes");
        files.insert(format!("{}.json", p.path), record);
        if let Some(reference) = &plan.reference {
            let into = folder(&p.path);
            let (pin, added) = gguf_documents(reference, into, req, &area, &mut files, &mut versions)?;
            filled.extend((!added.is_empty()).then_some((pin, added)));
        }
        ggufs_planned.push((p.file, component, plan));
    }
    let mut tensored: Vec<String> = Vec::new();
    let names = (planned.iter().map(|p| &p.component)).chain(ggufs_planned.iter().map(|g| &g.1));
    for name in names {
        if !tensored.contains(name) {
            tensored.push(name.clone());
        }
    }
    let sizes = |parts: &mut dyn Iterator<Item = (&Dtype, &[u64])>| -> u64 {
        parts.map(|(d, s)| d.byte_len(s).unwrap_or(0)).sum()
    };
    let stored = (planned.iter())
        .map(|p| {
            (
                &p.dtype,
                p.encoding.as_ref(),
                sizes(&mut p.parts.iter().map(|q| (&q.dtype, &q.shape[..]))),
            )
        })
        .chain(ggufs_planned.iter().flat_map(|g| &g.2.tensors).map(|o| {
            (
                &o.dtype,
                o.encoding.as_ref(),
                sizes(&mut o.parts.iter().map(|q| (&q.dtype, &q.shape[..]))),
            )
        }));
    stopped(req)?;
    let mut layout = layout(files, &tensored, &versions)?;
    for folder in &mut layout.components {
        if let Some(adapter) = adapters.remove(&folder.name) {
            (folder.config, folder.config_file) = (adapter.config, Some(lora::ADAPTER_CONFIG.into()));
        }
    }
    sink.begin(&main_variant(stored), &layout)?;
    let opened: Vec<File> = set
        .files
        .iter()
        .map(|f| {
            File::open(f).map_err(|e| crate::err::classify_io(format!("open {}", f.display()), e))
        })
        .collect::<Result<_>>()?;
    let gguf_tensors: usize = ggufs_planned.iter().map(|(_, _, p)| p.tensors.len()).sum();
    let total = (planned.len() + gguf_tensors) as u64;
    for (n, p) in planned.iter().enumerate() {
        stopped(req)?;
        progress("converting", n as u64, total);
        stopped(req)?;
        let what = format!("{}/{}", p.component, p.key);
        let mut readers = (p.parts.iter())
            .map(|part| {
                engine::TensorReader::new(&opened[part.read.file], &part.read, what.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let parts = (p.parts.iter().zip(&mut readers))
            .map(|(part, reader)| Part {
                role: &part.role,
                dtype: &part.dtype,
                shape: &part.shape,
                data: reader,
            })
            .collect();
        stopped(req)?;
        sink.put(
            &p.component,
            &p.key,
            &p.dtype,
            &p.shape,
            p.encoding.as_ref(),
            parts,
        )?;
    }
    for (file, component, plan) in &ggufs_planned {
        for o in &plan.tensors {
            stopped(req)?;
            let mut readers: Vec<gguf::Ranges> = (o.parts.iter())
                .map(|part| gguf::Ranges::new(&opened[*file], &part.ranges))
                .collect();
            let parts = (o.parts.iter().zip(&mut readers))
                .map(|(part, reader)| Part {
                    role: part.role,
                    dtype: &part.dtype,
                    shape: &part.shape,
                    data: reader,
                })
                .collect();
            let encoding = o.encoding.as_ref();
            stopped(req)?;
            sink.put(component, &o.key, &o.dtype, &o.shape, encoding, parts)?;
        }
    }
    let mut file_metadata: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let named = (recognized.claims.iter())
        .map(|c| (c.name.clone(), c.group))
        .chain(
            recognized
                .unclaimed
                .iter()
                .map(|g| (identity(&set.groups[*g]), *g)),
        );
    for (component, g) in named {
        let kept = (set.groups[g].metadata.iter())
            .filter(|(k, _)| !DECLARATIONS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()));
        file_metadata.entry(component).or_default().extend(kept);
    }
    file_metadata.retain(|_, m| !m.is_empty());
    let imported = Imported {
        normalized,
        converters,
        sources: pinned,
        filled,
        file_metadata,
    };
    // Reports completed conversion before publication: even the final progress callback
    // can refuse without committing a header. This is the final cooperative cancellation
    // boundary. Once commit starts, a
    // dropped future cannot promise to undo the writer's atomic finalization.
    progress("imported", total, total);
    stopped(req)?;
    sink.commit(imported.metadata())?;
    drop(opened);
    match std::fs::remove_dir_all(&area) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(crate::err::classify_io(
                format!("remove {}", area.display()),
                e,
            ));
        }
        _ => {}
    }
    Ok(imported)
}
