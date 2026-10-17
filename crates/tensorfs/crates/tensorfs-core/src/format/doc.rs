//! The three v2 documents and the tensor model. Each type parses through a `Wire` twin that
//! carries the `format` tag; validation happens once, in `TryFrom<Wire>`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    check_format, invalid, json, Dtype, Error, Result, Value, HEADER_FORMAT, HEADER_PATH,
    MANIFEST_FORMAT, REPO_FORMAT,
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

    /// The object's key in a store or static origin.
    pub fn key(&self) -> String {
        let h = &self.sha256;
        format!("blobs/{}/{}/{h}", &h[..2], &h[2..4])
    }
}

/// `tensorfs.manifest/2`: a checkpoint as a file tree. `model.cozytensors` is its header.
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

/// `cozytensors/2`: chunk objects and the tensors that live in them, by component.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "HeaderWire", into = "HeaderWire")]
pub struct Header {
    pub chunks: Vec<ObjectRef>,
    pub components: BTreeMap<String, Component>,
    pub metadata: Metadata,
}

#[derive(Clone, Serialize, Deserialize)]
struct HeaderWire {
    format: String,
    chunks: Vec<ObjectRef>,
    components: BTreeMap<String, Component>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Component {
    pub tensors: BTreeMap<String, Tensor>,
}

/// A tensor: its logical dtype and shape, and either its bytes or its encoded parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TensorWire", into = "TensorWire")]
pub struct Tensor {
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
        parts: BTreeMap<String, Part>,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct TensorWire {
    dtype: Dtype,
    shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<Location>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoding: Option<Encoding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parts: Option<BTreeMap<String, Part>>,
}

impl TryFrom<TensorWire> for Tensor {
    type Error = Error;
    fn try_from(w: TensorWire) -> Result<Self> {
        let body = match (w.data, w.encoding, w.parts) {
            (Some(at), None, None) => Body::Data(at),
            (None, Some(encoding), Some(parts)) if !parts.is_empty() => {
                if encoding.scheme.is_empty() {
                    return invalid("an encoding names no scheme");
                }
                if parts.keys().any(String::is_empty) {
                    return invalid("a part has an empty role");
                }
                Body::Encoded { encoding, parts }
            }
            _ => return invalid("a tensor has either `data`, or `encoding` and non-empty `parts`"),
        };
        Ok(Tensor {
            dtype: w.dtype,
            shape: w.shape,
            body,
        })
    }
}

impl From<Tensor> for TensorWire {
    fn from(t: Tensor) -> Self {
        let (data, encoding, parts) = match t.body {
            Body::Data(at) => (Some(at), None, None),
            Body::Encoded { encoding, parts } => (None, Some(encoding), Some(parts)),
        };
        TensorWire {
            dtype: t.dtype,
            shape: t.shape,
            data,
            encoding,
            parts,
        }
    }
}

/// `chunk` indexes `Header::chunks`. A range longer than the rest of its chunk continues at
/// offset 0 of the following chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Location {
    pub chunk: u64,
    pub offset: u64,
    pub length: u64,
}

/// One stored part of an encoded tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Part {
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Encoding {
    pub scheme: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: Metadata,
}

/// One chunk's share of a [`Location`]: bytes `start..start + length` of `chunk`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub chunk: usize,
    pub start: u64,
    pub length: u64,
}

impl TryFrom<HeaderWire> for Header {
    type Error = Error;
    fn try_from(w: HeaderWire) -> Result<Self> {
        check_format(&w.format, HEADER_FORMAT)?;
        let header = Header {
            chunks: w.chunks,
            components: w.components,
            metadata: w.metadata,
        };
        for (name, component) in &header.components {
            if !segment(name) {
                return invalid(format!(
                    "component name {name:?} is not one plain path segment"
                ));
            }
            for (key, t) in &component.tensors {
                if key.is_empty() {
                    return invalid(format!("component {name:?} has an empty tensor key"));
                }
                let at = |what: &str| format!("{name}/{key}{what}");
                match &t.body {
                    Body::Data(loc) => header.check(loc, t.dtype, &t.shape, &at(""))?,
                    Body::Encoded { parts, .. } => {
                        for (role, p) in parts {
                            header.check(
                                &p.at(),
                                p.dtype,
                                &p.shape,
                                &at(&format!(" part {role}")),
                            )?;
                        }
                    }
                }
            }
        }
        Ok(header)
    }
}

impl From<Header> for HeaderWire {
    fn from(h: Header) -> Self {
        HeaderWire {
            format: HEADER_FORMAT.into(),
            chunks: h.chunks,
            components: h.components,
            metadata: h.metadata,
        }
    }
}

impl Header {
    pub fn from_json(bytes: &[u8]) -> Result<Header> {
        json::read(bytes, HEADER_FORMAT)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        json::write(self)
    }

    pub fn tensor(&self, component: &str, key: &str) -> Option<&Tensor> {
        self.components.get(component)?.tensors.get(key)
    }

    /// The per-chunk spans of a location that [`Header::from_json`] accepted.
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

    fn check(&self, at: &Location, dtype: Dtype, shape: &[u64], what: &str) -> Result<()> {
        match dtype.byte_len(shape) {
            Some(n) if n == at.length => {}
            Some(n) => {
                return invalid(format!(
                    "{what}: length {} but {dtype:?} {shape:?} is {n} bytes",
                    at.length
                ))
            }
            None => {
                return invalid(format!(
                    "{what}: {dtype:?} {shape:?} is not a whole number of bytes"
                ))
            }
        }
        if at.length == 0 {
            return Ok(());
        }
        let Some(first) = self.chunks.get(at.chunk as usize) else {
            return invalid(format!(
                "{what}: chunk {} of {}",
                at.chunk,
                self.chunks.len()
            ));
        };
        let room: u128 = self.chunks[at.chunk as usize..]
            .iter()
            .map(|c| c.length as u128)
            .sum();
        if at.offset >= first.length || at.offset as u128 + at.length as u128 > room {
            return invalid(format!(
                "{what}: bytes {}+{} run past the chunks",
                at.offset, at.length
            ));
        }
        Ok(())
    }
}

/// `tensorfs.repo/2`: a model's releases, each mapping lanes to manifests. Kept at
/// `refs/<org>/<name>.json`; the last release is the latest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RepoWire", into = "RepoWire")]
pub struct Repo {
    pub name: String,
    pub releases: Vec<Release>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Release {
    pub version: String,
    pub lanes: BTreeMap<String, ObjectRef>,
}

#[derive(Clone, Serialize, Deserialize)]
struct RepoWire {
    format: String,
    name: String,
    releases: Vec<Release>,
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
        })
    }
}

impl From<Repo> for RepoWire {
    fn from(r: Repo) -> Self {
        RepoWire {
            format: REPO_FORMAT.into(),
            name: r.name,
            releases: r.releases,
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
    /// The repo's key in a store or static origin.
    pub fn key(&self) -> String {
        format!("refs/{}.json", self.name)
    }
    /// A lane's manifest in `version`, or in the latest release when `version` is `None`.
    pub fn lane(&self, version: Option<&str>, lane: &str) -> Option<&ObjectRef> {
        let release = match version {
            Some(v) => self.releases.iter().find(|r| r.version == v)?,
            None => self.releases.last()?,
        };
        release.lanes.get(lane)
    }
}

fn segment(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains(['/', '\\', '\0'])
}
