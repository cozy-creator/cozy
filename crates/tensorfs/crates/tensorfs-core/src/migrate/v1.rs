//! The last cozytensors/1 reader, kept only for the migration and deleted with it (xs-045 U12).
//! It reads a v1 store, or a directory holding a v1 manifest and its objects, and never writes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value as Json;

use crate::failure::{fail, Kind, Result};
use crate::format::{Dtype, ObjectRef};
use crate::sha256::hex_digest;

/// A v1 store's root, or a directory of `<sha256>` objects, read-only.
pub struct Source {
    root: PathBuf,
}

/// One v1 checkpoint and every repo, release and lane that names it.
#[derive(Debug, Clone)]
pub struct Lane {
    pub manifest: ObjectRef,
    pub names: Vec<String>,
}

pub struct Manifest {
    pub header: Option<ObjectRef>,
    pub files: Vec<(String, ObjectRef)>,
}

pub struct Header {
    /// Named configs, as their canonical JSON bytes.
    pub configs: Vec<(String, Vec<u8>)>,
    pub assets: Vec<Asset>,
    pub specs: Vec<Spec>,
    pub components: Vec<(String, Vec<Tensor>)>,
}

pub struct Asset {
    pub path: String,
    pub object: ObjectRef,
    pub segments: Vec<ObjectRef>,
}

/// An encoding spec: its digest (its identity in v1) and its roles with their carrier dtypes.
pub struct Spec {
    pub digest: String,
    pub roles: Vec<(String, Vec<Dtype>)>,
}

pub struct Tensor {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub spec: usize,
    pub parts: Vec<Part>,
}

pub struct Part {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub body: Body,
}

pub enum Body {
    Inline(Vec<u8>),
    /// Objects of the part's bytes, one per 64 MiB from its first byte.
    Objects(Vec<ObjectRef>),
}

impl Part {
    pub fn length(&self) -> u64 {
        match &self.body {
            Body::Inline(bytes) => bytes.len() as u64,
            Body::Objects(objects) => objects.iter().map(|o| o.length).sum(),
        }
    }
}

/// What a v1 spec becomes in v2.
pub enum Mapping {
    Plain,
    /// The v2 scheme and the part roles, v1 → v2.
    Scheme(&'static str, &'static [(&'static str, &'static str)]),
    /// A scheme v2 dropped, or a spec outside the v1 registry.
    Dropped,
}

const ROWWISE: &[(&str, &str)] = &[("data", "weight"), ("scale", "scale")];
const TENSORWISE: &[(&str, &str)] = &[
    ("data", "weight"),
    ("weight_scale", "scale"),
    ("input_scale", "input_scale"),
];
const NVFP4: &[(&str, &str)] = &[
    ("weight", "weight"),
    ("weight_scale", "scale"),
    ("weight_scale_2", "scale_2"),
    ("input_scale", "input_scale"),
];

/// The v1 registry's 11 specs by digest (tensorfs ≤ 0.11 `registry::seeds`).
static REGISTRY: [(&str, &str, Mapping); 11] = [
    (
        "1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f",
        "plain/1",
        Mapping::Plain,
    ),
    (
        "c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890",
        "fp8-rowwise/1",
        Mapping::Scheme("fp8-rowwise", ROWWISE),
    ),
    (
        "6773dfe384aeb380d8d32de9f0b384a59a3294a96da4f3f1c8b956e194e3ff2e",
        "fp8-rowwise/1 [out,1]",
        Mapping::Scheme("fp8-rowwise", ROWWISE),
    ),
    (
        "762e1d49dd11df1d9e99893213f232cf72797bf879aa6f07bc138ec86c04bac3",
        "fp8-scaled-scalar/1",
        Mapping::Scheme("fp8-tensorwise", TENSORWISE),
    ),
    (
        "7bede606ac5329020c2653eb04663b0d94d6c778e60234479509eddcce38240c",
        "fp8-scaled-scalar/1 weight-only",
        Mapping::Scheme("fp8-tensorwise", TENSORWISE),
    ),
    (
        "ac25d323baebfdf1fd0ce3fdfd9f26400ce57914fdc442d0ade8cb187aef183f",
        "int8-convrot256/1",
        Mapping::Scheme("int8-convrot", ROWWISE),
    ),
    (
        "f2e718c521634876830f9c2647bddfbbccafc744decc88adb345d33d73bf5b1a",
        "nvfp4-w4a4/1",
        Mapping::Scheme("nvfp4", NVFP4),
    ),
    (
        "2ce2ea887cd2880e2e55aa7cf4eaaeab5ec7a46d3bf648f20274822a286d3b11",
        "nvfp4-w4a4/1 +input_scale",
        Mapping::Scheme("nvfp4", NVFP4),
    ),
    (
        "19395cb7ce49dafe6787284b038337bffbb2b569d5aef05111c6c7c4caf86d0f",
        "nvfp4-w4a4/1 +pre_quant_scale",
        Mapping::Dropped,
    ),
    (
        "85416c825dbf6a8a253f439a2eaa3f31f9857a5a753b9f4a56b9fdbbe71661cd",
        "nvfp4-w4a4/1 +input_scale+pre_quant_scale",
        Mapping::Dropped,
    ),
    (
        "6e9117db61b068e19e3d0082921ddd72236682120501e62ae1c2cd6ca550d86f",
        "svdq-microscale/1",
        Mapping::Dropped,
    ),
];

impl Spec {
    /// The v1 alias (or the roles of a spec outside the registry) and what it becomes.
    pub fn mapping(&self) -> (String, &'static Mapping) {
        match REGISTRY.iter().find(|(d, ..)| *d == self.digest) {
            Some((_, alias, mapping)) => (alias.to_string(), mapping),
            None => {
                let roles: Vec<String> = (self.roles.iter())
                    .map(|(r, d)| {
                        format!(
                            "{r}:{}",
                            d.iter().map(Dtype::name).collect::<Vec<_>>().join("|")
                        )
                    })
                    .collect();
                let name = format!("unknown {} ({})", &self.digest[..12], roles.join(", "));
                (name, &Mapping::Dropped)
            }
        }
    }
}

const DTYPES: [Dtype; 12] = [
    Dtype::F64,
    Dtype::F32,
    Dtype::F16,
    Dtype::BF16,
    Dtype::F8_E4M3,
    Dtype::F8_E5M2,
    Dtype::I64,
    Dtype::I32,
    Dtype::I16,
    Dtype::I8,
    Dtype::U8,
    Dtype::BOOL,
];

fn bad<T>(detail: impl Into<String>) -> Result<T> {
    fail(Kind::Invalid, detail)
}

impl Source {
    pub fn new(root: impl Into<PathBuf>) -> Source {
        Source { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn find(&self, sha256: &str) -> Option<PathBuf> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let shard = |dir: &str, name: String| {
            self.root
                .join(dir)
                .join(&sha256[..2])
                .join(&sha256[2..4])
                .join(name)
        };
        [
            shard("blobs", sha256.to_owned()),
            shard("manifests", format!("{sha256}.json")),
            self.root.join(sha256),
        ]
        .into_iter()
        .find(|p| p.is_file())
    }

    /// Present with its length.
    pub fn has(&self, object: &ObjectRef) -> bool {
        let found = self.find(&object.sha256).and_then(|p| fs::metadata(p).ok());
        found.is_some_and(|m| m.len() == object.length)
    }

    /// An object's bytes, checked against its digest and length.
    pub fn read(&self, object: &ObjectRef) -> Result<Vec<u8>> {
        let Some(path) = self.find(&object.sha256) else {
            return fail(
                Kind::NotFound,
                format!(
                    "v1 object {} is not in {}",
                    object.sha256,
                    self.root.display()
                ),
            );
        };
        let bytes =
            fs::read(&path).or_else(|e| fail(Kind::Io, format!("{}: {e}", path.display())))?;
        if bytes.len() as u64 != object.length || hex_digest(&bytes) != object.sha256 {
            return fail(
                Kind::Integrity,
                format!(
                    "{} is not {} ({} bytes)",
                    path.display(),
                    object.sha256,
                    object.length
                ),
            );
        }
        Ok(bytes)
    }

    /// A manifest named by its digest alone.
    pub fn manifest_ref(&self, sha256: &str) -> Result<ObjectRef> {
        match self.find(sha256).and_then(|p| fs::metadata(p).ok()) {
            Some(m) => Ok(ObjectRef {
                sha256: sha256.to_owned(),
                length: m.len(),
            }),
            None => fail(
                Kind::NotFound,
                format!("no v1 manifest {sha256} in {}", self.root.display()),
            ),
        }
    }

    /// Every checkpoint the store's `repos/<org>/<name>.json` name, by manifest.
    pub fn lanes(&self) -> Result<Vec<Lane>> {
        let mut lanes: BTreeMap<ObjectRef, Vec<String>> = BTreeMap::new();
        let repos = self.root.join("repos");
        let list = |dir: &Path| -> Result<Vec<PathBuf>> {
            let entries =
                fs::read_dir(dir).or_else(|e| fail(Kind::Io, format!("{}: {e}", dir.display())))?;
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            Ok(paths)
        };
        for org in list(&repos)? {
            for path in list(&org)? {
                let text = fs::read(&path)
                    .or_else(|e| fail(Kind::Io, format!("{}: {e}", path.display())))?;
                let repo: Json = serde_json::from_slice(&text)
                    .or_else(|e| bad(format!("{}: {e}", path.display())))?;
                let name = format!(
                    "{}/{}",
                    str_at(&repo, &["repo", "org"])?,
                    str_at(&repo, &["repo", "name"])?
                );
                for c in array(&repo, "checkpoints") {
                    let names = lanes.entry(object(&c["manifest"])?).or_default();
                    names.push(name.clone());
                }
                for r in array(&repo, "releases") {
                    let version = str_at(&r, &["version"])?;
                    for l in array(&r, "lanes") {
                        let names = lanes.entry(object(&l["manifest"])?).or_default();
                        names.push(format!("{name}@{version} {}", str_at(&l, &["lane"])?));
                    }
                }
            }
        }
        Ok(lanes
            .into_iter()
            .map(|(manifest, mut names)| {
                names.sort();
                names.dedup();
                Lane { manifest, names }
            })
            .collect())
    }
}

fn array(doc: &Json, field: &str) -> Vec<Json> {
    doc.get(field)
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default()
}

fn str_at(doc: &Json, path: &[&str]) -> Result<String> {
    let found = path
        .iter()
        .try_fold(doc, |d, f| d.get(f))
        .and_then(Json::as_str);
    match found {
        Some(s) => Ok(s.to_owned()),
        None => bad(format!("v1 document has no {}", path.join("."))),
    }
}

fn object(doc: &Json) -> Result<ObjectRef> {
    Ok(ObjectRef {
        sha256: str_at(doc, &["sha256"])?,
        length: doc
            .get("length")
            .and_then(Json::as_u64)
            .map_or_else(|| bad("v1 ref has no length"), Ok)?,
    })
}

/// `{"entries": [{"path", "kind", "blob"}]}`: the header entry and ordinary files.
pub fn manifest(bytes: &[u8]) -> Result<Manifest> {
    let doc: Json = serde_json::from_slice(bytes).or_else(|e| bad(format!("v1 manifest: {e}")))?;
    let mut out = Manifest {
        header: None,
        files: Vec::new(),
    };
    for e in array(&doc, "entries") {
        let (path, kind, blob) = (
            str_at(&e, &["path"])?,
            str_at(&e, &["kind"])?,
            object(&e["blob"])?,
        );
        match kind.as_str() {
            "cozytensors" => out.header = Some(blob),
            "file" => out.files.push((path, blob)),
            other => {
                return fail(
                    Kind::Unsupported,
                    format!("v1 manifest entry {path:?} is a {other:?}"),
                )
            }
        }
    }
    Ok(out)
}

/// A CBOR item: the restricted subset v1 headers use.
enum Item<'a> {
    Uint(u64),
    Bytes(&'a [u8]),
    Text(&'a str),
    Array(Vec<Item<'a>>),
}

struct Cbor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cbor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        match self.bytes.get(self.at..self.at + n) {
            Some(s) => {
                self.at += n;
                Ok(s)
            }
            None => bad("v1 header: truncated"),
        }
    }

    fn item(&mut self) -> Result<Item<'a>> {
        let first = self.take(1)?[0];
        let n = match first & 31 {
            v @ 0..=23 => v as u64,
            v @ 24..=27 => self
                .take(1 << (v - 24))?
                .iter()
                .fold(0, |n, b| n << 8 | *b as u64),
            _ => return bad("v1 header: indefinite or reserved length"),
        };
        Ok(match first >> 5 {
            0 => Item::Uint(n),
            2 => Item::Bytes(self.take(n as usize)?),
            3 => match std::str::from_utf8(self.take(n as usize)?) {
                Ok(s) => Item::Text(s),
                Err(_) => return bad("v1 header: text is not UTF-8"),
            },
            4 => Item::Array((0..n).map(|_| self.item()).collect::<Result<_>>()?),
            major => return bad(format!("v1 header: CBOR major type {major}")),
        })
    }
}

impl<'a> Item<'a> {
    fn uint(&self) -> Result<u64> {
        match self {
            Item::Uint(n) => Ok(*n),
            _ => bad("v1 header: expected an integer"),
        }
    }
    fn text(&self) -> Result<&'a str> {
        match self {
            Item::Text(s) => Ok(s),
            _ => bad("v1 header: expected text"),
        }
    }
    fn list(&self, len: Option<usize>) -> Result<&[Item<'a>]> {
        match self {
            Item::Array(v) if len.is_none_or(|n| n == v.len()) => Ok(v),
            _ => bad("v1 header: unexpected array shape"),
        }
    }
    fn dtype(&self) -> Result<Dtype> {
        match DTYPES.get(self.uint()? as usize) {
            Some(d) => Ok(d.clone()),
            None => bad("v1 header: dtype code out of range"),
        }
    }
    fn shape(&self) -> Result<Vec<u64>> {
        self.list(None)?.iter().map(Item::uint).collect()
    }
    fn object(&self) -> Result<ObjectRef> {
        match self.list(Some(2))? {
            [Item::Bytes(d), n] if d.len() == 32 => Ok(ObjectRef {
                sha256: d.iter().map(|b| format!("{b:02x}")).collect(),
                length: n.uint()?,
            }),
            _ => bad("v1 header: malformed object ref"),
        }
    }
}

/// A `cozytensors/1` header: `[format, configs, assets, encodings, components]`.
pub fn header(bytes: &[u8]) -> Result<Header> {
    let mut d = Cbor { bytes, at: 0 };
    let first = d.take(1)?[0];
    if first != 0x85 || d.item()?.text()? != "cozytensors/1" {
        return fail(Kind::Format, "not a cozytensors/1 header");
    }
    let configs = (d.item()?.list(None)?.iter())
        .map(|c| match c.list(Some(2))? {
            [name, Item::Bytes(json)] => Ok((name.text()?.to_owned(), json.to_vec())),
            _ => bad("v1 header: malformed config"),
        })
        .collect::<Result<_>>()?;
    let assets = (d.item()?.list(None)?.iter())
        .map(|a| match a.list(Some(5))? {
            [name, Item::Bytes(digest), length, _media, segments] if digest.len() == 32 => {
                Ok(Asset {
                    path: name.text()?.to_owned(),
                    object: ObjectRef {
                        sha256: digest.iter().map(|b| format!("{b:02x}")).collect(),
                        length: length.uint()?,
                    },
                    segments: segments
                        .list(None)?
                        .iter()
                        .map(Item::object)
                        .collect::<Result<_>>()?,
                })
            }
            _ => bad("v1 header: malformed asset"),
        })
        .collect::<Result<_>>()?;
    // Encodings: each spec's identity is the digest of its exact bytes.
    let head = d.take(1)?[0];
    let count = match head {
        0x80..=0x97 => (head - 0x80) as u64,
        0x98 => d.take(1)?[0] as u64,
        0x99 => d.take(2)?.iter().fold(0, |n, b| n << 8 | *b as u64),
        _ => return bad("v1 header: malformed encodings"),
    };
    let mut specs = Vec::new();
    for _ in 0..count {
        let start = d.at;
        let spec = d.item()?;
        let digest = hex_digest(&bytes[start..d.at]);
        let roles = match spec.list(Some(4))? {
            [_, _, roles, _] => (roles.list(None)?.iter())
                .map(|r| {
                    let [name, carrier, _] = r.list(Some(3))? else {
                        unreachable!()
                    };
                    let dtypes = match carrier.list(None)? {
                        [_, set] => set
                            .list(None)?
                            .iter()
                            .map(Item::dtype)
                            .collect::<Result<_>>()?,
                        _ => vec![],
                    };
                    Ok((name.text()?.to_owned(), dtypes))
                })
                .collect::<Result<_>>()?,
            _ => unreachable!(),
        };
        specs.push(Spec { digest, roles });
    }
    let components = (d.item()?.list(None)?.iter())
        .map(|c| {
            let [name, tensors] = c.list(Some(2))? else {
                unreachable!()
            };
            let tensors = (tensors.list(None)?.iter())
                .map(|t| {
                    let [key, dtype, shape, spec, parts] = t.list(Some(5))? else {
                        unreachable!()
                    };
                    let spec = spec.uint()? as usize;
                    if spec >= specs.len() {
                        return bad("v1 header: encoding index out of range");
                    }
                    let parts = (parts.list(None)?.iter())
                        .map(|p| {
                            let [role, dtype, shape, body] = p.list(Some(4))? else {
                                unreachable!()
                            };
                            let body = match body {
                                Item::Bytes(b) => Body::Inline(b.to_vec()),
                                Item::Array(v) => Body::Objects(
                                    v.iter().map(Item::object).collect::<Result<_>>()?,
                                ),
                                _ => return bad("v1 header: malformed part body"),
                            };
                            Ok(Part {
                                role: role.text()?.to_owned(),
                                dtype: dtype.dtype()?,
                                shape: shape.shape()?,
                                body,
                            })
                        })
                        .collect::<Result<_>>()?;
                    Ok(Tensor {
                        key: key.text()?.to_owned(),
                        dtype: dtype.dtype()?,
                        shape: shape.shape()?,
                        spec,
                        parts,
                    })
                })
                .collect::<Result<_>>()?;
            Ok((name.text()?.to_owned(), tensors))
        })
        .collect::<Result<_>>()?;
    if d.at != bytes.len() {
        return bad("v1 header: trailing bytes");
    }
    Ok(Header {
        configs,
        assets,
        specs,
        components,
    })
}
