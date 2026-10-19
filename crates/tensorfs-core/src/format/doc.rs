//! The three documents and the tensor model. The manifest and repo parse through a `Wire` twin
//! that carries the `format` tag, the header through `cbor`; validation happens once, in
//! `TryFrom<Wire>` and [`Header::new`].

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::layout::ALIGN;
use super::{
    cbor, check_format, decode, invalid, json, Config, Dtype, Error, PartBytes, Result, Value,
    HEADER_PATH, MANIFEST_FORMAT, REPO_FORMAT,
};

/// Free-form metadata, and a scheme's params.
pub type Metadata = BTreeMap<String, Value>;

/// An immutable object: `{sha256, length}`, stored at `blobs/<aa>/<bb>/<sha256>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "RefWire")]
pub struct ObjectRef {
    pub sha256: String,
    pub length: u64,
}

#[derive(Deserialize)]
struct RefWire {
    sha256: String,
    length: u64,
}

impl TryFrom<RefWire> for ObjectRef {
    type Error = Error;
    fn try_from(w: RefWire) -> Result<Self> {
        let hex = |c: u8| c.is_ascii_digit() || (b'a'..=b'f').contains(&c);
        if w.sha256.len() != 64 || !w.sha256.bytes().all(hex) {
            return invalid(format!(
                "sha256 {:?} is not 64 lowercase hex digits",
                w.sha256
            ));
        }
        Ok(ObjectRef {
            sha256: w.sha256,
            length: w.length,
        })
    }
}

impl ObjectRef {
    pub fn of(bytes: &[u8]) -> ObjectRef {
        let digest = Sha256::digest(bytes);
        let sha256 = digest.iter().map(|b| format!("{b:02x}")).collect();
        ObjectRef {
            sha256,
            length: bytes.len() as u64,
        }
    }

    /// `sha256:<hex>`, the object's name where a digest is named alone.
    pub fn id(&self) -> String {
        format!("sha256:{}", self.sha256)
    }

    /// The object's key in a store or static origin.
    pub fn key(&self) -> String {
        let h = &self.sha256;
        format!("blobs/{}/{}/{h}", &h[..2], &h[2..4])
    }
}

/// `tensorfs.manifest/1`: a checkpoint as a file tree. `model.cozytensors` is its header.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "ManifestWire", into = "ManifestWire")]
pub struct Manifest {
    pub files: BTreeMap<String, ObjectRef>,
    pub metadata: Metadata,
}

#[derive(Clone, Serialize, Deserialize)]
struct ManifestWire {
    format: String,
    files: BTreeMap<String, ObjectRef>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata: Metadata,
}

impl TryFrom<ManifestWire> for Manifest {
    type Error = Error;
    fn try_from(w: ManifestWire) -> Result<Self> {
        check_format(&w.format, MANIFEST_FORMAT)?;
        for path in w.files.keys() {
            if !path.split('/').all(segment) {
                return invalid(format!(
                    "file path {path:?} is not a relative path of plain segments"
                ));
            }
        }
        Ok(Manifest {
            files: w.files,
            metadata: w.metadata,
        })
    }
}

impl From<Manifest> for ManifestWire {
    fn from(m: Manifest) -> Self {
        ManifestWire {
            format: MANIFEST_FORMAT.into(),
            files: m.files,
            metadata: m.metadata,
        }
    }
}

impl Manifest {
    pub fn from_json(bytes: &[u8]) -> Result<Manifest> {
        json::read(bytes, MANIFEST_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }
    pub fn header(&self) -> Option<&ObjectRef> {
        self.files.get(HEADER_PATH)
    }
}

/// `cozytensors/1`: one variant of a model, self-contained: its pipeline, its components (class,
/// config, tensors, assets), the chunk objects the tensors live in, and loading metadata.
/// Components, each component's tensors and each tensor's parts are in construction order (the
/// order a model builds and fills them), which the header's bytes and digest carry. Lookup by
/// name goes through an index built by [`Header::new`].
#[derive(Debug, Clone, Default)]
pub struct Header {
    pub chunks: Vec<ObjectRef>,
    pipeline: Option<Pipeline>,
    components: Vec<Component>,
    /// Loading metadata: `source`, `filled`, per-file safetensors `__metadata__`.
    pub metadata: Metadata,
    /// Component name → (its index, tensor key → tensor index).
    index: HashMap<String, (usize, HashMap<String, usize>)>,
}

impl PartialEq for Header {
    fn eq(&self, other: &Header) -> bool {
        (&self.chunks, &self.pipeline, &self.components, &self.metadata)
            == (&other.chunks, &other.pipeline, &other.components, &other.metadata)
    }
}

/// What `model_index.json` held.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pipeline {
    /// `<library>.<Class>` (`diffusers.StableDiffusionXLPipeline`), `""` when unknown.
    pub class: String,
    pub config: Config,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Component {
    pub name: String,
    /// `<library>.<Class>` (`transformers.CLIPTextModel`), `""` when unknown.
    pub class: String,
    pub config: Config,
    pub tensors: Vec<Tensor>,
    /// Run data in the component's folder (`tokenizer.json`, `vocab.json`, ...), by path
    /// relative to it.
    pub assets: BTreeMap<String, ObjectRef>,
    /// The config's file name, when it isn't `config.json`.
    pub config_file: Option<String>,
}

impl Component {
    /// The config's file name: [`Component::config_file`] or `config.json`.
    pub fn config_name(&self) -> &str {
        self.config_file.as_deref().unwrap_or(CONFIG_FILE)
    }
}

/// A component config's file name unless `config_file` says otherwise.
pub const CONFIG_FILE: &str = "config.json";

/// A tensor: its key, logical dtype and shape, and either its bytes or its encoded parts.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub body: Body,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// Plain: the bytes at `data` are the tensor, little-endian, row-major.
    Data(Location),
    /// Encoded: `encoding` names the scheme that turns `parts` into the logical tensor.
    Encoded {
        encoding: Encoding,
        parts: Vec<Part>,
    },
}

impl Tensor {
    /// An encoded tensor's part by role.
    pub fn part(&self, role: &str) -> Option<&Part> {
        match &self.body {
            Body::Encoded { parts, .. } => parts.iter().find(|p| p.role == role),
            Body::Data(_) => None,
        }
    }

    /// Every stored range, in order: a plain tensor's data (role `""`), or each part.
    pub fn locations(&self) -> Vec<(&str, Location)> {
        match &self.body {
            Body::Data(at) => vec![("", *at)],
            Body::Encoded { parts, .. } => {
                parts.iter().map(|p| (p.role.as_str(), p.at())).collect()
            }
        }
    }
}

/// `chunk` indexes `Header::chunks`. A range longer than the rest of its chunk continues at
/// offset 0 of the following chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Location {
    pub chunk: u64,
    pub offset: u64,
    pub length: u64,
}

/// One stored part of an encoded tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub chunk: u64,
    pub offset: u64,
    pub length: u64,
}

impl Part {
    pub fn at(&self) -> Location {
        Location {
            chunk: self.chunk,
            offset: self.offset,
            length: self.length,
        }
    }
}

/// A scheme name from the spec's registry and its explicit params.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Encoding {
    pub scheme: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: Metadata,
}

/// `nvfp4 {block_size: 16, global_scale: "multiply", ...}`: params in key order.
impl std::fmt::Display for Encoding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {}", self.scheme, Value::Map(self.params.clone()))
    }
}

/// One chunk's share of a [`Location`]: bytes `start..start + length` of `chunk`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub chunk: usize,
    pub start: u64,
    pub length: u64,
}

impl Header {
    /// A header with `components` in the given (construction) order. Refuses repeated names,
    /// keys or roles, locations outside `chunks`, asset names that are not paths, and a
    /// `config_file` of `config.json` or that names an asset.
    pub fn new(
        chunks: Vec<ObjectRef>,
        pipeline: Option<Pipeline>,
        components: Vec<Component>,
        metadata: Metadata,
    ) -> Result<Header> {
        let mut header = Header {
            chunks,
            pipeline,
            components: Vec::new(),
            metadata,
            index: HashMap::new(),
        };
        // rooms[i]: the bytes of chunks[i..].
        let mut rooms: Vec<u128> = header.chunks.iter().rev()
            .scan(0u128, |sum, c| {
                *sum += c.length as u128;
                Some(*sum)
            })
            .collect();
        rooms.reverse();
        for (ci, component) in components.iter().enumerate() {
            let name = &component.name;
            if !segment(name) {
                return invalid(format!(
                    "component name {name:?} is not one plain path segment"
                ));
            }
            if let Some(file) = &component.config_file {
                if file == CONFIG_FILE || !path(file) {
                    return invalid(format!("component {name:?}: config_file {file:?}"));
                }
            }
            if let Some(asset) = (component.assets.keys())
                .find(|a| !path(a) || *a == component.config_name())
            {
                return invalid(format!(
                    "component {name:?}: asset {asset:?} is not a path or is its config file"
                ));
            }
            let mut keys = HashMap::with_capacity(component.tensors.len());
            for (ti, t) in component.tensors.iter().enumerate() {
                let key = &t.key;
                if key.is_empty() || keys.insert(key.clone(), ti).is_some() {
                    return invalid(format!(
                        "component {name:?}: tensor key {key:?} is empty or repeated"
                    ));
                }
                match &t.body {
                    Body::Data(loc) => {
                        header.check(&rooms, loc, &t.dtype, &t.shape, || format!("{name}/{key}"))?
                    }
                    Body::Encoded { encoding, parts } => {
                        if encoding.scheme.is_empty() || parts.is_empty() {
                            return invalid(format!("{name}/{key}: no scheme or no parts"));
                        }
                        for (i, p) in parts.iter().enumerate() {
                            let role = &p.role;
                            if role.is_empty() || parts[..i].iter().any(|o| &o.role == role) {
                                return invalid(format!(
                                    "{name}/{key}: part role {role:?} is empty or repeated"
                                ));
                            }
                            let what = || format!("{name}/{key} part {role}");
                            header.check(&rooms, &p.at(), &p.dtype, &p.shape, what)?;
                        }
                    }
                }
            }
            if header.index.insert(name.clone(), (ci, keys)).is_some() {
                return invalid(format!("component {name:?} is repeated"));
            }
        }
        header.components = components;
        Ok(header)
    }

    /// A header from its bytes (spec §9), read strictly.
    pub fn from_cbor(bytes: &[u8]) -> Result<Header> {
        cbor::read(bytes)
    }
    /// The header's deterministic bytes.
    pub fn to_cbor(&self) -> Result<Vec<u8>> {
        cbor::write(self)
    }

    /// What `model_index.json` held; `None` for a checkpoint of one model.
    pub fn pipeline(&self) -> Option<&Pipeline> {
        self.pipeline.as_ref()
    }

    /// The components, in construction order.
    pub fn components(&self) -> &[Component] {
        &self.components
    }

    /// Every component's assets: (component, asset path, object).
    pub fn assets(&self) -> impl Iterator<Item = (&str, &str, &ObjectRef)> {
        self.components.iter().flat_map(|c| {
            (c.assets.iter()).map(move |(path, r)| (c.name.as_str(), path.as_str(), r))
        })
    }

    /// The header's closure besides itself: its chunks, then every component's assets.
    pub fn objects(&self) -> impl Iterator<Item = &ObjectRef> {
        self.chunks.iter().chain(self.assets().map(|(_, _, r)| r))
    }

    /// The distinct encodings, in order of first citation by the tensors (the header's table).
    pub fn encodings(&self) -> Vec<&Encoding> {
        let mut out: Vec<&Encoding> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_, t) in self.tensors() {
            if let Body::Encoded { encoding, .. } = &t.body {
                if seen.insert(encoding) {
                    out.push(encoding);
                }
            }
        }
        out
    }

    pub fn component(&self, name: &str) -> Option<&Component> {
        self.index.get(name).map(|(i, _)| &self.components[*i])
    }

    pub fn tensor(&self, component: &str, key: &str) -> Option<&Tensor> {
        let (ci, ti) = self.position(component, key)?;
        Some(&self.components[ci].tensors[ti])
    }

    /// Where a tensor is in construction order: (its component's index, its index there).
    pub fn position(&self, component: &str, key: &str) -> Option<(usize, usize)> {
        let (ci, keys) = self.index.get(component)?;
        Some((*ci, *keys.get(key)?))
    }

    /// Every tensor with its component's name, in construction order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.components
            .iter()
            .flat_map(|c| c.tensors.iter().map(move |t| (c.name.as_str(), t)))
    }

    /// The per-chunk spans of a location that [`Header::new`] accepted.
    pub fn spans(&self, at: &Location) -> Vec<Span> {
        let mut out = Vec::new();
        let (mut chunk, mut start, mut left) = (at.chunk as usize, at.offset, at.length);
        while left > 0 {
            let length = left.min(self.chunks[chunk].length - start);
            out.push(Span {
                chunk,
                start,
                length,
            });
            (chunk, start, left) = (chunk + 1, 0, left - length);
        }
        out
    }

    /// A location's bytes, gathered from whole chunks that `chunk` supplies by index.
    pub fn bytes<B: AsRef<[u8]>>(
        &self,
        at: &Location,
        mut chunk: impl FnMut(usize) -> Result<B>,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(at.length as usize);
        for s in self.spans(at) {
            let bytes = chunk(s.chunk)?;
            let bytes = bytes.as_ref();
            if bytes.len() as u64 != self.chunks[s.chunk].length {
                return invalid(format!(
                    "chunk {} is {} bytes, its ref says {}",
                    s.chunk,
                    bytes.len(),
                    self.chunks[s.chunk].length
                ));
            }
            out.extend_from_slice(&bytes[s.start as usize..(s.start + s.length) as usize]);
        }
        Ok(out)
    }

    /// A tensor's logical bytes: a plain tensor's data, or its parts through the reference decode.
    pub fn read<B: AsRef<[u8]>>(
        &self,
        component: &str,
        key: &str,
        mut chunk: impl FnMut(usize) -> Result<B>,
    ) -> Result<Vec<u8>> {
        let Some(t) = self.tensor(component, key) else {
            return invalid(format!("no tensor {component}/{key}"));
        };
        match &t.body {
            Body::Data(at) => self.bytes(at, &mut chunk),
            Body::Encoded { encoding, parts } => {
                let bytes = parts
                    .iter()
                    .map(|p| self.bytes(&p.at(), &mut chunk))
                    .collect::<Result<Vec<_>>>()?;
                let views = parts
                    .iter()
                    .zip(&bytes)
                    .map(|(p, b)| {
                        (
                            p.role.as_str(),
                            PartBytes {
                                dtype: &p.dtype,
                                shape: &p.shape,
                                bytes: b,
                            },
                        )
                    })
                    .collect();
                decode(&t.dtype, &t.shape, encoding, &views)
            }
        }
    }

    /// Sizes known dtypes; an unknown dtype's `length` stands as written. `rooms[i]` is the
    /// bytes of `chunks[i..]`.
    fn check(
        &self,
        rooms: &[u128],
        at: &Location,
        dtype: &Dtype,
        shape: &[u64],
        what: impl Fn() -> String,
    ) -> Result<()> {
        match (dtype.bits(), dtype.byte_len(shape)) {
            (None, _) => {}
            (_, Some(n)) if n == at.length => {}
            (_, Some(n)) => {
                return invalid(format!(
                    "{}: length {} but {dtype} {shape:?} is {n} bytes",
                    what(),
                    at.length
                ))
            }
            (_, None) => {
                return invalid(format!(
                    "{}: {dtype} {shape:?} overflows or is not whole bytes",
                    what()
                ))
            }
        }
        if at.length == 0 {
            return Ok(());
        }
        let Some(first) = self.chunks.get(at.chunk as usize) else {
            return invalid(format!(
                "{}: chunk {} of {}",
                what(),
                at.chunk,
                self.chunks.len()
            ));
        };
        if !at.offset.is_multiple_of(ALIGN) {
            return invalid(format!("{}: offset {} is not 4 KiB aligned", what(), at.offset));
        }
        let room = rooms[at.chunk as usize];
        if at.offset >= first.length || at.offset as u128 + at.length as u128 > room {
            return invalid(format!(
                "{}: bytes {}+{} run past the chunks",
                what(),
                at.offset,
                at.length
            ));
        }
        Ok(())
    }
}

/// `refs/<org>/<name>.json`, ASCII-lowercased: repo names are case-insensitive, one identity.
pub fn repo_key(name: &str) -> String {
    format!("refs/{}.json", name.to_ascii_lowercase())
}

/// `tensorfs.repo/1`: a model's releases, each mapping lanes to manifests, and its unreleased
/// checkpoints. Kept at [`repo_key`]; releases are in publication order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "RepoWire", into = "RepoWire")]
pub struct Repo {
    pub name: String,
    pub releases: Vec<Release>,
    /// Uploaded checkpoints no release cites yet. GC roots, like release lanes.
    pub checkpoints: Vec<ObjectRef>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    pub lanes: BTreeMap<String, ObjectRef>,
    /// Skipped by an unversioned lookup; an explicit version still resolves it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub yanked: bool,
    /// The release's expected-revision counter: each change to it increments this.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub revision: u64,
}

fn is_false(b: &bool) -> bool {
    !b
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

#[derive(Clone, Serialize, Deserialize)]
struct RepoWire {
    format: String,
    name: String,
    releases: Vec<Release>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    checkpoints: Vec<ObjectRef>,
}

impl TryFrom<RepoWire> for Repo {
    type Error = Error;
    fn try_from(w: RepoWire) -> Result<Self> {
        check_format(&w.format, REPO_FORMAT)?;
        match w.name.split_once('/') {
            Some((org, name)) if segment(org) && segment(name) => {}
            _ => return invalid(format!("repo name {:?} is not <org>/<name>", w.name)),
        }
        let mut seen = std::collections::HashSet::new();
        for r in &w.releases {
            if r.version.is_empty() || !seen.insert(&r.version) {
                return invalid(format!(
                    "release version {:?} is empty or repeated",
                    r.version
                ));
            }
            if r.lanes.keys().any(String::is_empty) {
                return invalid(format!("release {:?} has an empty lane name", r.version));
            }
        }
        Ok(Repo {
            name: w.name,
            releases: w.releases,
            checkpoints: w.checkpoints,
        })
    }
}

impl From<Repo> for RepoWire {
    fn from(r: Repo) -> Self {
        RepoWire {
            format: REPO_FORMAT.into(),
            name: r.name,
            releases: r.releases,
            checkpoints: r.checkpoints,
        }
    }
}

impl Repo {
    pub fn from_json(bytes: &[u8]) -> Result<Repo> {
        json::read(bytes, REPO_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }
    /// The repo's key in a store or static origin: [`repo_key`] of its name.
    pub fn key(&self) -> String {
        repo_key(&self.name)
    }
    /// Release `version`, yanked or not; with `None`, the newest release that is not yanked.
    pub fn release(&self, version: Option<&str>) -> Option<&Release> {
        match version {
            Some(v) => self.releases.iter().find(|r| r.version == v),
            None => self.releases.iter().rev().find(|r| !r.yanked),
        }
    }
    /// A lane's manifest in [`Repo::release`]`(version)`.
    pub fn lane(&self, version: Option<&str>, lane: &str) -> Option<&ObjectRef> {
        self.release(version)?.lanes.get(lane)
    }
    /// Every manifest the repo keeps alive: each release's lanes (yanked too) and each checkpoint.
    pub fn roots(&self) -> impl Iterator<Item = &ObjectRef> {
        let lanes = self.releases.iter().flat_map(|r| r.lanes.values());
        lanes.chain(&self.checkpoints)
    }
}

fn segment(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains(['/', '\\', '\0'])
}

/// One or more segments joined by `/`.
fn path(s: &str) -> bool {
    s.split('/').all(segment)
}
