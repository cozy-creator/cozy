//! The last cozytensors/1 (v1) reader, for A11 alone and deleted with it (owner, 2026-10-20: no
//! public 0.12, so production's v1 lanes go straight to D10). A v1 lane becomes D10 tensors
//! citing its bytes where they already are: a part's objects (one per 64 MiB) become consecutive
//! chunks. Inline parts, tiny and at unaligned header offsets, which D10 cannot cite, are copied
//! byte for byte into one pack per lane, each at a 4 KiB boundary: the lane's only new chunk.
//! Names and construction order follow U12's rules (D4); specs map to D10 schemes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::{json, Value as Json};

use crate::failure::{fail, Kind, Result};
use crate::format::decode::{check_metadata, PartMetadata};
use crate::format::layout::ALIGN;
use crate::format::{Body, Dtype, Encoding, GlobalScale, Location, NibbleOrder, ObjectRef, Part, ScaleLayout, Tensor};
use crate::import::{constructor_order, order, tables};
use crate::sha256::hex_digest;

/// `{"entries": [{path, kind, blob}]}`: the header and the lane's files.
pub struct Manifest {
    pub header: ObjectRef,
    pub files: Vec<(String, ObjectRef)>,
}

/// A v1 lane in D10's terms: chunks, and components of tensors (renamed, in construction
/// order); its files by D10 path; the inline parts' pack, when it has any; and what changed.
pub struct Lane {
    pub chunks: Vec<ObjectRef>,
    pub pack: Option<(ObjectRef, Vec<u8>)>,
    pub components: Vec<(String, Vec<Tensor>)>,
    pub files: BTreeMap<String, File>,
    pub report: Json,
}

/// A file of the lane: a v1 object, or a config's bytes.
pub enum File {
    Object(ObjectRef),
    Bytes(Vec<u8>),
}

/// The v1 manifest in `bytes`; `None` when it is not one (a D10 or v2 document).
pub fn manifest(bytes: &[u8]) -> Result<Option<Manifest>> {
    let Ok(doc) = serde_json::from_slice::<Json>(bytes) else { return Ok(None) };
    let Some(entries) = doc.get("entries").and_then(Json::as_array) else { return Ok(None) };
    let (mut header, mut files) = (None, Vec::new());
    for e in entries {
        let text = |k: &str| e.get(k).and_then(Json::as_str).map(str::to_owned);
        let blob = &e["blob"];
        let (Some(path), Some(kind), Some(sha256), Some(length)) =
            (text("path"), text("kind"), blob["sha256"].as_str(), blob["length"].as_u64())
        else {
            return bad("v1 manifest: malformed entry");
        };
        let object = ObjectRef { sha256: sha256.to_owned(), length };
        match kind.as_str() {
            "cozytensors" => header = Some(object),
            "file" => files.push((path, object)),
            other => return bad(format!("v1 manifest entry {path:?} is a {other:?}")),
        }
    }
    match header {
        Some(header) => Ok(Some(Manifest { header, files })),
        None => bad("v1 manifest: no header"),
    }
}

fn bad<T>(detail: impl Into<String>) -> Result<T> {
    fail(Kind::Invalid, detail)
}

/// A v1 spec by the digest of its bytes (its identity in v1): the D10 scheme and v1 → D10 roles.
enum Mapping {
    Plain,
    Scheme(&'static str, &'static [(&'static str, &'static str)]),
    Dropped,
}

const ROWWISE: &[(&str, &str)] = &[("data", "weight"), ("scale", "scale")];
const TENSORWISE: &[(&str, &str)] = &[("data", "weight"), ("weight_scale", "scale"), ("input_scale", "input_scale")];
const NVFP4: &[(&str, &str)] =
    &[("weight", "weight"), ("weight_scale", "scale"), ("weight_scale_2", "scale_2"), ("input_scale", "input_scale")];

/// v1's registry (tensorfs ≤ 0.11 `registry::seeds`) and the early H3 mxfp8 spec, as U12 mapped
/// them.
static REGISTRY: [(&str, &str, Mapping); 12] = [
    ("7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd", "mxfp8 (E8M0 scale as U8)", Mapping::Scheme("mxfp8", ROWWISE)),
    ("1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f", "plain/1", Mapping::Plain),
    ("c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890", "fp8-rowwise/1", Mapping::Scheme("fp8.rowwise", ROWWISE)),
    ("6773dfe384aeb380d8d32de9f0b384a59a3294a96da4f3f1c8b956e194e3ff2e", "fp8-rowwise/1 [out,1]", Mapping::Scheme("fp8.rowwise", ROWWISE)),
    ("762e1d49dd11df1d9e99893213f232cf72797bf879aa6f07bc138ec86c04bac3", "fp8-scaled-scalar/1", Mapping::Scheme("fp8.tensorwise", TENSORWISE)),
    ("7bede606ac5329020c2653eb04663b0d94d6c778e60234479509eddcce38240c", "fp8-scaled-scalar/1 weight-only", Mapping::Scheme("fp8.tensorwise", TENSORWISE)),
    ("ac25d323baebfdf1fd0ce3fdfd9f26400ce57914fdc442d0ade8cb187aef183f", "int8-convrot256/1", Mapping::Scheme("int8.convrot", ROWWISE)),
    ("f2e718c521634876830f9c2647bddfbbccafc744decc88adb345d33d73bf5b1a", "nvfp4-w4a4/1", Mapping::Scheme("nvfp4", NVFP4)),
    ("2ce2ea887cd2880e2e55aa7cf4eaaeab5ec7a46d3bf648f20274822a286d3b11", "nvfp4-w4a4/1 +input_scale", Mapping::Scheme("nvfp4", NVFP4)),
    ("19395cb7ce49dafe6787284b038337bffbb2b569d5aef05111c6c7c4caf86d0f", "nvfp4-w4a4/1 +pre_quant_scale", Mapping::Dropped),
    ("85416c825dbf6a8a253f439a2eaa3f31f9857a5a753b9f4a56b9fdbbe71661cd", "nvfp4-w4a4/1 +input_scale+pre_quant_scale", Mapping::Dropped),
    ("6e9117db61b068e19e3d0082921ddd72236682120501e62ae1c2cd6ca550d86f", "svdq-microscale/1", Mapping::Dropped),
];

fn mapping(digest: &str) -> (&'static str, &'static Mapping) {
    match REGISTRY.iter().find(|(d, ..)| *d == digest) {
        Some((_, alias, m)) => (alias, m),
        None => ("unknown", &Mapping::Dropped),
    }
}

const DTYPES: [Dtype; 12] = [
    Dtype::F64, Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::F8_E4M3, Dtype::F8_E5M2,
    Dtype::I64, Dtype::I32, Dtype::I16, Dtype::I8, Dtype::U8, Dtype::BOOL,
];

/// The restricted CBOR v1 headers use.
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
        let Some(s) = self.bytes.get(self.at..self.at + n) else { return bad("v1 header: truncated") };
        self.at += n;
        Ok(s)
    }

    fn item(&mut self) -> Result<Item<'a>> {
        let first = self.take(1)?[0];
        let n = match first & 31 {
            v @ 0..=23 => v as u64,
            v @ 24..=27 => self.take(1 << (v - 24))?.iter().fold(0, |n, b| n << 8 | *b as u64),
            _ => return bad("v1 header: indefinite or reserved length"),
        };
        Ok(match first >> 5 {
            0 => Item::Uint(n),
            2 => Item::Bytes(self.take(n as usize)?),
            3 => Item::Text(std::str::from_utf8(self.take(n as usize)?).or_else(|_| bad("v1 header: text is not UTF-8"))?),
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
        DTYPES.get(self.uint()? as usize).cloned().map_or_else(|| bad("v1 header: dtype code out of range"), Ok)
    }
    fn shape(&self) -> Result<Vec<u64>> {
        self.list(None)?.iter().map(Item::uint).collect()
    }
    fn object(&self) -> Result<ObjectRef> {
        match self.list(Some(2))? {
            [Item::Bytes(d), n] if d.len() == 32 => Ok(ObjectRef { sha256: hex(d), length: n.uint()? }),
            _ => bad("v1 header: malformed object ref"),
        }
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

struct Header<'a> {
    configs: Vec<(String, &'a [u8])>,
    assets: Vec<(String, ObjectRef, Vec<ObjectRef>)>,
    specs: Vec<String>,
    components: Vec<(String, Vec<V1Tensor>)>,
}

struct V1Tensor {
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    spec: usize,
    parts: Vec<V1Part>,
}

struct V1Part {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    body: V1Body,
}

enum V1Body {
    /// Bytes `offset..offset + length` of the header object itself.
    Inline { offset: u64, length: u64 },
    /// Objects of the part's bytes, one per 64 MiB from its first byte.
    Objects(Vec<ObjectRef>),
}

/// `[format, configs, assets, encodings, components]`, tagged `cozytensors/1`.
fn header(bytes: &[u8]) -> Result<Header<'_>> {
    let mut d = Cbor { bytes, at: 0 };
    if d.take(1)?[0] != 0x85 || d.item()?.text()? != "cozytensors/1" {
        return fail(Kind::Format, "not a v1 cozytensors/1 header");
    }
    let configs = (d.item()?.list(None)?.iter())
        .map(|c| match c.list(Some(2))? {
            [name, Item::Bytes(json)] => Ok((name.text()?.to_owned(), *json)),
            _ => bad("v1 header: malformed config"),
        })
        .collect::<Result<_>>()?;
    let assets = (d.item()?.list(None)?.iter())
        .map(|a| match a.list(Some(5))? {
            [name, Item::Bytes(digest), length, _media, segments] if digest.len() == 32 => Ok((
                name.text()?.to_owned(),
                ObjectRef { sha256: hex(digest), length: length.uint()? },
                segments.list(None)?.iter().map(Item::object).collect::<Result<_>>()?,
            )),
            _ => bad("v1 header: malformed asset"),
        })
        .collect::<Result<_>>()?;
    // A spec's identity is the digest of its exact bytes.
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
        d.item()?;
        specs.push(hex_digest(&bytes[start..d.at]));
    }
    let base = bytes.as_ptr() as usize;
    let components = (d.item()?.list(None)?.iter())
        .map(|c| {
            let [name, tensors] = c.list(Some(2))? else { return bad("v1 header: malformed component") };
            let tensors = (tensors.list(None)?.iter())
                .map(|t| {
                    let [key, dtype, shape, spec, parts] = t.list(Some(5))? else { return bad("v1 header: malformed tensor") };
                    let spec = spec.uint()? as usize;
                    if spec >= specs.len() {
                        return bad("v1 header: encoding index out of range");
                    }
                    let parts = (parts.list(None)?.iter())
                        .map(|p| {
                            let [role, dtype, shape, body] = p.list(Some(4))? else { return bad("v1 header: malformed part") };
                            let body = match body {
                                Item::Bytes(b) => V1Body::Inline { offset: (b.as_ptr() as usize - base) as u64, length: b.len() as u64 },
                                Item::Array(v) => V1Body::Objects(v.iter().map(Item::object).collect::<Result<_>>()?),
                                _ => return bad("v1 header: malformed part body"),
                            };
                            Ok(V1Part { role: role.text()?.to_owned(), dtype: dtype.dtype()?, shape: shape.shape()?, body })
                        })
                        .collect::<Result<_>>()?;
                    Ok(V1Tensor { key: key.text()?.to_owned(), dtype: dtype.dtype()?, shape: shape.shape()?, spec, parts })
                })
                .collect::<Result<_>>()?;
            Ok((name.text()?.to_owned(), tensors))
        })
        .collect::<Result<_>>()?;
    if d.at != bytes.len() {
        return bad("v1 header: trailing bytes");
    }
    Ok(Header { configs, assets, specs, components })
}

/// The objects of a v1 header's assets: what a lane reads besides its manifest and header.
pub fn assets(raw: &[u8]) -> Result<Vec<ObjectRef>> {
    Ok(header(raw)?.assets.into_iter().flat_map(|(_, _, segments)| segments).collect())
}

/// The lane `m` names, its header's bytes `raw`, in D10's terms. Refuses a dropped or unknown
/// spec, a part D10's decoder would refuse, and an asset in several objects (it would need new
/// bytes).
pub fn lane(m: &Manifest, raw: &[u8]) -> Result<Lane> {
    if ObjectRef::of(raw) != m.header {
        return fail(Kind::Integrity, format!("v1 header is not {}", m.header.sha256));
    }
    let h = header(raw)?;
    let mut names = Names::of(&h);
    let files = files(m, &h, &mut names)?;
    let order = ordering(&h, &names);
    let (mut chunks, mut runs): (Vec<ObjectRef>, HashMap<Vec<ObjectRef>, u64>) = (Vec::new(), HashMap::new());
    let (mut pack, mut pack_at): (Vec<u8>, Option<u64>) = (Vec::new(), None);
    let mut place = |body: &V1Body| -> Location {
        match body {
            V1Body::Inline { length: 0, .. } => Location::default(),
            V1Body::Inline { offset, length } => {
                let chunk = *pack_at.get_or_insert_with(|| {
                    chunks.push(ObjectRef { sha256: String::new(), length: 0 });
                    chunks.len() as u64 - 1
                });
                pack.resize(pack.len().next_multiple_of(ALIGN as usize), 0);
                let at = pack.len() as u64;
                pack.extend_from_slice(&raw[*offset as usize..(offset + length) as usize]);
                Location { chunk, offset: at, length: *length }
            }
            V1Body::Objects(objects) => {
                let length = objects.iter().map(|o| o.length).sum();
                if length == 0 {
                    return Location::default();
                }
                let chunk = *runs.entry(objects.clone()).or_insert_with(|| {
                    chunks.extend(objects.iter().cloned());
                    (chunks.len() - objects.len()) as u64
                });
                Location { chunk, offset: 0, length }
            }
        }
    };
    let (mut components, mut schemes): (Vec<(String, Vec<Tensor>)>, BTreeMap<&str, u64>) = (Vec::new(), BTreeMap::new());
    for (c, i) in order {
        let (v1_name, tensors) = &h.components[c];
        let t = &tensors[i];
        let (alias, mapping) = mapping(&h.specs[t.spec]);
        *schemes.entry(alias).or_default() += 1;
        let key = names.key(v1_name, &t.key);
        let what = format!("{v1_name}/{}", t.key);
        let body = match mapping {
            Mapping::Plain => match t.parts.as_slice() {
                [p] if p.dtype == t.dtype && p.shape == t.shape => Body::Data(place(&p.body)),
                _ => return bad(format!("{what}: a plain tensor is not one part of its dtype and shape")),
            },
            Mapping::Scheme(scheme, roles) => {
                let mut parts = Vec::new();
                for p in &t.parts {
                    let (role, dtype, shape) = role(t, scheme, roles, p).map_err(|e| invalid(format!("{what}: {e}")))?;
                    let at = place(&p.body);
                    parts.push(Part { role, dtype, shape, chunk: at.chunk, offset: at.offset, length: at.length });
                }
                let encoding = encoding(t, scheme);
                let meta: Vec<PartMetadata> = (parts.iter())
                    .map(|p| PartMetadata { role: &p.role, dtype: &p.dtype, shape: &p.shape, len: p.length as usize })
                    .collect();
                check_metadata(&t.dtype, &t.shape, &encoding, &meta).map_err(|e| invalid(format!("{what}: {encoding}: {e}")))?;
                Body::Encoded { encoding, parts }
            }
            Mapping::Dropped => return fail(Kind::Unsupported, format!("{what}: spec {alias} has no D10 scheme")),
        };
        let component = names.component(v1_name);
        match components.last_mut() {
            Some((name, ts)) if *name == component => ts.push(Tensor { key, dtype: t.dtype.clone(), shape: t.shape.clone(), body }),
            _ => components.push((component, vec![Tensor { key, dtype: t.dtype.clone(), shape: t.shape.clone(), body }])),
        }
    }
    let inline = (h.components.iter().flat_map(|c| &c.1).flat_map(|t| &t.parts))
        .filter(|p| matches!(p.body, V1Body::Inline { .. }))
        .count();
    let pack = pack_at.map(|i| {
        let at = ObjectRef::of(&pack);
        chunks[i as usize] = at.clone();
        (at, pack)
    });
    let report = json!({"v1": {"header": m.header.id(), "schemes": schemes, "inline_parts": inline,
        "pack": pack.as_ref().map(|(at, _)| at.id()), "pack_bytes": pack.as_ref().map_or(0, |(at, _)| at.length), "chunks": chunks.len()}, "renames": names.report()});
    Ok(Lane { chunks, pack, components, files, report })
}

fn invalid(detail: String) -> crate::failure::Failure {
    crate::failure::Failure { kind: Kind::Invalid, detail }
}

/// A v1 part's D10 role, dtype and shape: the same bytes, named as the scheme does.
fn role(t: &V1Tensor, scheme: &str, roles: &[(&str, &str)], p: &V1Part) -> std::result::Result<(String, Dtype, Vec<u64>), String> {
    let Some((_, role)) = roles.iter().find(|(v1, _)| *v1 == p.role) else {
        return Err(format!("no D10 role for {}", p.role));
    };
    Ok(match (scheme, *role, t.shape.as_slice()) {
        // v1's packed `U8 [R, C/2]` is `F4 [R, C]`.
        ("nvfp4", "weight", [rows, cols]) if p.dtype == Dtype::U8 && p.shape == [*rows, cols / 2] => {
            (role.to_string(), Dtype::F4, vec![*rows, *cols])
        }
        ("nvfp4", "weight", _) => return Err(format!("nvfp4 weight {:?}", p.shape)),
        // The early mxfp8 stored its E8M0 scales as U8.
        ("mxfp8", "scale", _) if p.dtype == Dtype::U8 => (role.to_string(), Dtype::F8_E8M0, p.shape.clone()),
        _ => (role.to_string(), p.dtype.clone(), p.shape.clone()),
    })
}

/// `t`'s encoding under `scheme`, every scale in the dtype v1 stored it in (U12's mapping).
fn encoding(t: &V1Tensor, scheme: &str) -> Encoding {
    let scale = |role: &str| t.parts.iter().find(|p| p.role == role).map_or(Dtype::F32, |p| p.dtype.clone());
    match scheme {
        "fp8.rowwise" => Encoding::fp8_rowwise(0, &scale("scale")),
        "fp8.tensorwise" => Encoding::fp8_tensorwise(&scale("weight_scale")),
        "int8.convrot" => Encoding::int8_convrot(256, &scale("scale")),
        "mxfp8" => Encoding::mxfp8(ScaleLayout::RowMajor),
        // v1's nvfp4 is ModelOpt's layout.
        "nvfp4" => Encoding::nvfp4(GlobalScale::Multiply, NibbleOrder::LowFirst, ScaleLayout::RowMajor),
        other => unreachable!("v1 maps to no {other}"),
    }
}

/// The lane's files at D10 paths: the manifest's files, single-object assets, and the inline
/// configs at their Diffusers or Transformers paths (H3's one `model` config split per component).
fn files(m: &Manifest, h: &Header, names: &mut Names) -> Result<BTreeMap<String, File>> {
    let mut out = BTreeMap::new();
    let add = |out: &mut BTreeMap<String, File>, path: String, file: File| match out.get(&path) {
        Some(File::Object(a)) if matches!(&file, File::Object(b) if a == b) => Ok(()),
        Some(_) => bad(format!("two v1 files land at {path}")),
        None => Ok(drop(out.insert(path, file))),
    };
    for (path, object) in &m.files {
        add(&mut out, path.clone(), File::Object(object.clone()))?;
    }
    for (path, object, segments) in &h.assets {
        match segments.as_slice() {
            [one] if one == object => add(&mut out, path.clone(), File::Object(one.clone()))?,
            _ => return fail(Kind::Unsupported, format!("asset {path} is {} objects: it needs new bytes", segments.len())),
        }
    }
    let components: BTreeSet<&str> = h.components.iter().map(|c| c.0.as_str()).collect();
    for (name, bytes) in &h.configs {
        let path = match name.as_str() {
            "model_index" => "model_index.json".to_owned(),
            "scheduler" => "scheduler/scheduler_config.json".to_owned(),
            "model" => {
                let doc: Json = serde_json::from_slice(bytes).unwrap_or_default();
                let per_component = doc.as_object().filter(|map| {
                    !components.contains("model") && map.len() == components.len() && map.keys().all(|k| components.contains(k.as_str()))
                });
                if let Some(map) = per_component {
                    for (c, config) in map {
                        let split = serde_json::to_vec(config).or_else(|e| bad(e.to_string()))?;
                        add(&mut out, format!("{}/config.json", names.component(c)), File::Bytes(split))?;
                    }
                    names.rules.push("config model: one config.json per component".into());
                    continue;
                }
                // A lane of one component other than `model`: the config is that component's.
                match components.iter().collect::<Vec<_>>().as_slice() {
                    [only] if **only != "model" => format!("{}/config.json", names.component(only)),
                    _ => "config.json".to_owned(),
                }
            }
            c if components.contains(c) => format!("{}/config.json", names.component(c)),
            other => format!("{other}.json"),
        };
        if matches!(out.get(&path), Some(File::Object(_))) {
            names.rules.push(format!("config {name}: the lane's own {path} kept"));
            continue;
        }
        add(&mut out, path, File::Bytes(bytes.to_vec()))?;
    }
    Ok(out)
}

/// v1's names for H3's upstream folders; the tables name each folder's canonical component.
const V1_FOLDERS: [(&str, &str); 3] = [("fl2va_dit", "FL2VA/transformer"), ("ref2va_dit", "Ref2VA/transformer"), ("video_vae", "vae")];
/// LoRA factor suffixes, as v1 and the tables spell them (PEFT).
const LORA: [&str; 3] = [".lora_A.weight", ".lora_B.weight", ".alpha"];

/// D4's canonical names for one v1 lane (U12's rules).
#[derive(Default)]
struct Names {
    components: BTreeMap<String, String>,
    keys: HashMap<(String, String), String>,
    dropped: BTreeSet<(String, String)>,
    rules: Vec<String>,
}

impl Names {
    fn of(h: &Header) -> Names {
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
            let Some((table, renames, drop)) = table_keys(&name, &keys) else { continue };
            if let Some((from, to)) = renames.first() {
                n.rules.push(format!("{component}: {} keys renamed by {table}, e.g. {from} → {to}", renames.len()));
            }
            for (from, to) in renames {
                n.keys.insert((component.clone(), from), to);
            }
            for key in drop {
                n.rules.push(format!("{component}: {key} dropped by {table}"));
                n.dropped.insert((component.clone(), key));
            }
        }
        n
    }

    fn component(&self, c: &str) -> String {
        self.components.get(c).cloned().unwrap_or_else(|| c.to_owned())
    }

    fn key(&self, c: &str, k: &str) -> String {
        self.keys.get(&(c.to_owned(), k.to_owned())).cloned().unwrap_or_else(|| k.to_owned())
    }

    fn report(&self) -> Json {
        json!({"components": self.components, "rules": self.rules})
    }
}

/// A v1 component's canonical name when it differs: an H3 folder's, or a LoRA's one table
/// component holding every module it adapts.
fn canonical_component(component: &str, keys: &[&str]) -> Option<String> {
    if let Some((_, folder)) = V1_FOLDERS.iter().find(|(v1, _)| *v1 == component) {
        let tabled = (tables().iter().flat_map(|t| &t.components)).find_map(|c| c.folders.get(*folder));
        return tabled.cloned().or_else(|| (!folder.contains('/')).then(|| folder.to_string()));
    }
    let stems: BTreeSet<&str> = keys.iter().map(|k| LORA.iter().find_map(|s| k.strip_suffix(s))).collect::<Option<_>>()?;
    let mut claims = (tables().iter().filter(|t| t.lora.is_some())).flat_map(|t| &t.components).filter(|c| {
        let modules: BTreeSet<&str> = c.rows.iter().map(|r| r.target.as_str()).collect();
        !stems.is_empty() && stems.iter().all(|s| modules.contains(s))
    });
    match (claims.next(), claims.next()) {
        (Some(c), None) if c.name != component => Some(c.name.clone()),
        _ => None,
    }
}

/// The table, renames and drops that make `keys` a table component's canonical keys.
fn table_keys(component: &str, keys: &[&str]) -> Option<(String, Vec<(String, String)>, Vec<String>)> {
    for t in tables().iter().filter(|t| t.lora.is_none()) {
        for c in t.components.iter().filter(|c| c.name == component || c.folders.values().any(|f| f == component)) {
            let targets: BTreeSet<&str> = c.rows.iter().map(|r| r.target.as_str()).collect();
            let drop: Vec<String> = keys.iter().filter(|k| c.drop.iter().any(|d| d == *k)).map(|k| k.to_string()).collect();
            let missing: Vec<&str> = keys.iter().copied().filter(|k| !targets.contains(k) && !drop.iter().any(|d| d == k)).collect();
            // A prefix v1 left off (SDXL's `text_model.`).
            let prefix = match missing.first() {
                None => "",
                Some(m) => match targets.iter().find_map(|t| t.strip_suffix(m).filter(|p| p.ends_with('.'))) {
                    Some(p) => p,
                    None => continue,
                },
            };
            if missing.iter().all(|k| targets.contains(format!("{prefix}{k}").as_str())) {
                let renames = missing.iter().map(|k| (k.to_string(), format!("{prefix}{k}"))).collect();
                return Some((t.name.clone(), renames, drop));
            }
        }
    }
    None
}

/// `(component, tensor)` indices in construction order. Components keep v1's order; a
/// component v1 sorted lexically takes its class's construction order, unknown keys keeping
/// v1's; dropped keys leave.
fn ordering(h: &Header, names: &Names) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (c, (name, tensors)) in h.components.iter().enumerate() {
        let kept: Vec<usize> = (0..tensors.len()).filter(|i| !names.dropped.contains(&(name.clone(), tensors[*i].key.clone()))).collect();
        let v1_keys: Vec<&str> = tensors.iter().map(|t| t.key.as_str()).collect();
        let mut order = kept.clone();
        if sorted(&v1_keys) {
            let keys: Vec<String> = kept.iter().map(|i| names.key(name, &tensors[*i].key)).collect();
            let indexed: Vec<(&str, usize)> = keys.iter().map(String::as_str).zip(0..).collect();
            let model = model_class(h, name).or_else(|| probed_class(&keys));
            order = constructor_order(model.as_deref(), &indexed).order.iter().map(|i| kept[*i]).collect();
        }
        out.extend(order.into_iter().map(|i| (c, i)));
    }
    out
}

/// Whether v1 wrote these keys lexically sorted rather than in construction order (U12's test:
/// few descents, and weights not consistently before their biases).
fn sorted(keys: &[&str]) -> bool {
    let n = keys.len();
    let descents = keys.windows(2).filter(|w| w[0] > w[1]).count();
    let at: HashMap<&str, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let (mut pairs, mut weight_first) = (0, 0);
    for (i, k) in keys.iter().enumerate() {
        if let Some(w) = k.strip_suffix(".bias").and_then(|m| at.get(format!("{m}.weight").as_str())) {
            pairs += 1;
            weight_first += usize::from(*w < i);
        }
    }
    let lexical = match (n, descents) {
        (0..=2, _) => false,
        (_, 0) => true,
        (0..=19, _) => false,
        _ => descents <= 3 || descents * 20 < n,
    };
    lexical && !(pairs > 0 && weight_first == pairs)
}

/// A component's model class from v1's inline configs: its own, its entry in a `model` config
/// holding every component's, or a single component's `model` config.
fn model_class(h: &Header, component: &str) -> Option<String> {
    let config = |name: &str| serde_json::from_slice::<Json>(h.configs.iter().find(|c| c.0 == name)?.1).ok();
    let class = |doc: &Json| doc.get("_class_name").or_else(|| doc.get("architectures")?.get(0)).and_then(Json::as_str).map(str::to_owned);
    if let Some(doc) = config(component) {
        return class(&doc);
    }
    let model = config("model")?;
    class(&model[component]).or_else(|| (h.components.len() == 1).then(|| class(&model))?)
}

/// An adapter's class: the one whose construction order knows all but a tenth of its modules.
fn probed_class(keys: &[String]) -> Option<String> {
    const FACTORS: [&str; 5] = ["lora_A", "lora_B", "lora_down", "lora_up", "alpha"];
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
    let (best, unknown) = order::classes().map(|c| (c, constructor_order(Some(c), &indexed).unknown.len())).min_by_key(|(_, u)| *u)?;
    (unknown * 10 < keys.len()).then(|| best.to_owned())
}
