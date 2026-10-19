//! GGUF, ggml's container (v3, ggml `docs/gguf.md`): typed metadata (the KV), tensor infos, then
//! each tensor's bytes at an aligned offset.
//!
//! - A GGUF imports as one component. A ggml block type is the scheme `gguf.<type>`
//!   (`{block_size}`) whose one part, `blocks`, holds llama.cpp's struct bytes as written
//!   (spec §6); a plain ggml type is a plain tensor. The logical shape is `ne` reversed (or
//!   ComfyUI-GGUF's `comfy.gguf.orig_shape.<name>`) and an encoded tensor's dtype is `F32`, what
//!   ggml decodes to. ggml's `nvfp4` keeps its per-tensor `.scale` and `.input_scale` tensors as
//!   parts.
//! - LLM files take their Transformers keys from their arch's name map (`gguf_names.json`,
//!   llama.cpp's `tensor_mapping.py` reversed): expert stacks split per expert and llama's
//!   converter Q/K row permutation is undone, both lossless on block rows. Other files keep
//!   their keys.
//! - The KV and the tensor table become `<file>.json` beside where the GGUF was; `export::gguf`
//!   rebuilds the file from it.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::sync::OnceLock;

use platform::durable::FileAt;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::err::{refuse, Code, Result};
use crate::format::decode::gguf::{blocks_shape, Type};
use crate::format::{Dtype, Encoding};

use super::remote::{self, Access, Member};

/// The dtype of an unblocked ggml type, stored as a plain tensor.
pub fn plain(ty: Type) -> Option<Dtype> {
    Some(match ty.name {
        "f32" => Dtype::F32,
        "f16" => Dtype::F16,
        "bf16" => Dtype::BF16,
        "f64" => Dtype::F64,
        "i8" => Dtype::I8,
        "i16" => Dtype::I16,
        "i32" => Dtype::I32,
        "i64" => Dtype::I64,
        _ => return None,
    })
}

// ------------------------------------------------------------------------------ the KV

/// One KV entry as the record holds it: key, GGUF type and value. An array's value is
/// `[element type, [items]]`; a float is its shortest decimal that reads back to its bits, or
/// those bits as `"0x…"` when it is not finite.
pub type Kv = (String, String, Json);

const KINDS: [&str; 13] = [
    "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64",
];
/// The integer types: name, bytes, signed.
const INTS: [(&str, usize, bool); 8] = [
    ("u8", 1, false),
    ("i8", 1, true),
    ("u16", 2, false),
    ("i16", 2, true),
    ("u32", 4, false),
    ("i32", 4, true),
    ("u64", 8, false),
    ("i64", 8, true),
];
/// Nesting of arrays in arrays.
const DEPTH: u32 = 8;

fn kind_of(tag: u32) -> Result<&'static str> {
    match KINDS.get(tag as usize) {
        Some(k) => Ok(k),
        None => refuse(Code::WRONG_TYPE, format!("GGUF value type {tag}")),
    }
}

fn tag(kind: &str) -> Result<u32> {
    match KINDS.iter().position(|k| *k == kind) {
        Some(t) => Ok(t as u32),
        None => refuse(Code::WRONG_TYPE, format!("GGUF value type {kind:?}")),
    }
}

fn f32_json(v: f32) -> Json {
    let short: f64 = v.to_string().parse().expect("a float prints as a float");
    match (v.is_finite(), (short as f32).to_bits() == v.to_bits()) {
        (true, true) => short.into(),
        (true, false) => (v as f64).into(),
        (false, _) => format!("0x{:08x}", v.to_bits()).into(),
    }
}

fn f64_json(v: f64) -> Json {
    match v.is_finite() {
        true => v.into(),
        false => format!("0x{:016x}", v.to_bits()).into(),
    }
}

/// A KV value's bytes (without its type tag), from the record.
fn write_value(kind: &str, v: &Json, out: &mut Vec<u8>) -> Result<()> {
    let bad = || refuse(Code::WRONG_TYPE, format!("GGUF {kind} value {v}"));
    let bits = v
        .as_str()
        .and_then(|s| u64::from_str_radix(s.strip_prefix("0x")?, 16).ok());
    if let Some((_, n, signed)) = INTS.iter().find(|(k, ..)| *k == kind) {
        let x = (v.as_i64().map(i128::from)).or(v.as_u64().map(i128::from));
        let (lo, hi) = match signed {
            true => (-(1i128 << (8 * n - 1)), (1i128 << (8 * n - 1)) - 1),
            false => (0, (1i128 << (8 * n)) - 1),
        };
        let Some(x) = x.filter(|x| (lo..=hi).contains(x)) else {
            return bad();
        };
        out.extend(&(x as u64).to_le_bytes()[..*n]);
        return Ok(());
    }
    match kind {
        "f32" => match (v.as_f64(), bits) {
            (Some(x), _) => out.extend((x as f32).to_le_bytes()),
            (_, Some(b)) if b <= u32::MAX.into() => out.extend((b as u32).to_le_bytes()),
            _ => return bad(),
        },
        "f64" => match (v.as_f64(), bits) {
            (Some(x), _) => out.extend(x.to_le_bytes()),
            (_, Some(b)) => out.extend(b.to_le_bytes()),
            _ => return bad(),
        },
        "bool" => match v.as_bool() {
            Some(b) => out.push(b as u8),
            None => return bad(),
        },
        "string" => match v.as_str() {
            Some(text) => string(text, out),
            None => return bad(),
        },
        "array" => {
            let Some([Json::String(inner), Json::Array(items)]) = v.as_array().map(Vec::as_slice)
            else {
                return bad();
            };
            out.extend(tag(inner)?.to_le_bytes());
            out.extend((items.len() as u64).to_le_bytes());
            for item in items {
                write_value(inner, item, out)?;
            }
        }
        _ => return bad(),
    }
    Ok(())
}

fn string(s: &str, out: &mut Vec<u8>) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

// ------------------------------------------------------------------------------ the container

/// Bytes read by offset: a local file, or a remote one by HTTP range.
pub trait At {
    fn size(&self) -> u64;
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;
}

impl At for File {
    fn size(&self) -> u64 {
        self.metadata().map_or(0, |m| m.len())
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_exact_at(buf, offset)
            .map_err(|e| crate::err::classify_io("read a GGUF", e))
    }
}

/// A remote member read by range, its length as listed.
pub struct Remote<'a> {
    pub member: &'a Member,
    pub access: &'a Access,
}

impl At for Remote<'_> {
    fn size(&self) -> u64 {
        self.member.length.unwrap_or(0)
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let got = remote::read_range(self.member, offset, offset + buf.len() as u64, self.access)?;
        if got.len() != buf.len() {
            return refuse(
                Code::SHORT_READ,
                format!("{}: {} of {} bytes", self.member.path, got.len(), buf.len()),
            );
        }
        buf.copy_from_slice(&got);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Info {
    pub name: String,
    pub ne: Vec<u64>,
    pub ty: Type,
    /// From the start of the data section.
    pub offset: u64,
    pub nbytes: u64,
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub kv: Vec<Kv>,
    pub tensors: Vec<Info>,
    pub alignment: u64,
    /// The data section's first byte in the file.
    pub data: u64,
}

impl Gguf {
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.kv.iter().find(|(k, ..)| k == key).map(|(.., v)| v)
    }
}

/// The header may be this large (a 256 k-token vocabulary is ~10 MB).
const HEADER_CAP: u64 = 1 << 30;
const CHUNK: u64 = 1 << 20;
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml's `GGML_MAX_DIMS`.
const MAX_DIMS: u32 = 4;

/// Reads the header incrementally through `src`, fetching a MiB at a time.
struct Cursor<'a> {
    src: &'a dyn At,
    len: u64,
    buf: Vec<u8>,
    pos: u64,
}

impl Cursor<'_> {
    fn take(&mut self, n: u64) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.len);
        let Some(end) = end else {
            return refuse(
                Code::CARRIER_TRUNCATED,
                format!("GGUF ends at {} inside its header", self.len),
            );
        };
        if end > HEADER_CAP {
            return refuse(
                Code::SIZE_CAP,
                format!("GGUF header past {HEADER_CAP} bytes"),
            );
        }
        let have = self.buf.len() as u64;
        if end > have {
            let to = end.max(have + CHUNK).min(self.len);
            let mut more = vec![0; (to - have) as usize];
            self.src.read_at(have, &mut more)?;
            self.buf.extend(more);
        }
        let at = self.pos as usize;
        self.pos = end;
        Ok(&self.buf[at..end as usize])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        match String::from_utf8(self.take(n)?.to_vec()) {
            Ok(s) => Ok(s),
            Err(_) => refuse(Code::WRONG_TYPE, "GGUF string is not UTF-8"),
        }
    }

    fn value(&mut self, kind: &str, depth: u32) -> Result<Json> {
        if let Some((_, n, signed)) = INTS.iter().find(|(k, ..)| *k == kind) {
            let mut b = [0u8; 8];
            b[..*n].copy_from_slice(self.take(*n as u64)?);
            let shift = 64 - 8 * *n as u32;
            return Ok(match signed {
                true => ((i64::from_le_bytes(b) << shift) >> shift).into(),
                false => u64::from_le_bytes(b).into(),
            });
        }
        Ok(match kind {
            "f32" => f32_json(f32::from_le_bytes(self.take(4)?.try_into().expect("4"))),
            "f64" => f64_json(f64::from_le_bytes(self.take(8)?.try_into().expect("8"))),
            "bool" => match self.take(1)?[0] {
                v @ (0 | 1) => (v == 1).into(),
                v => return refuse(Code::WRONG_TYPE, format!("GGUF bool {v}")),
            },
            "string" => self.string()?.into(),
            _ => {
                if depth == DEPTH {
                    return refuse(Code::DEPTH_CAP, "GGUF arrays nest too deep");
                }
                let inner = kind_of(self.u32()?)?;
                let n = self.u64()?;
                // Every element takes a byte at least: a count past the file is a lie.
                if n > self.len - self.pos {
                    return refuse(Code::COUNT_CAP, format!("GGUF array of {n}"));
                }
                let items = (0..n).map(|_| self.value(inner, depth + 1));
                Json::Array(vec![
                    inner.into(),
                    Json::Array(items.collect::<Result<_>>()?),
                ])
            }
        })
    }
}

fn pad(n: u64, alignment: u64) -> u64 {
    n.next_multiple_of(alignment)
}

/// The header of the GGUF in `src`: KV, tensor infos and where the data section starts. Every
/// tensor must lie in the file at the offset llama.cpp's own reader requires (each one after the
/// last, padded to the alignment).
pub fn read(src: &dyn At) -> Result<Gguf> {
    let mut c = Cursor {
        src,
        len: src.size(),
        buf: Vec::new(),
        pos: 0,
    };
    if c.take(4)? != b"GGUF" {
        return refuse(Code::UNKNOWN_FORMAT, "not a GGUF (magic)");
    }
    let version = c.u32()?;
    if version != 3 {
        let what = match version.swap_bytes() {
            1..=3 => "a big-endian GGUF".to_owned(),
            _ => format!("GGUF version {version}; only 3"),
        };
        return refuse(Code::UNKNOWN_FORMAT, what);
    }
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    if n_tensors > c.len || n_kv > c.len {
        return refuse(Code::COUNT_CAP, format!("{n_tensors} tensors, {n_kv} KV"));
    }
    let mut kv: Vec<Kv> = Vec::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let kind = kind_of(c.u32()?)?;
        let value = c.value(kind, 0)?;
        if kv.iter().any(|(k, ..)| *k == key) {
            return refuse(Code::DUPLICATE_KEY, format!("GGUF KV {key} twice"));
        }
        kv.push((key, kind.to_owned(), value));
    }
    let alignment = alignment(&kv)?;
    let split = kv.iter().find(|(k, ..)| k == "split.count");
    if let Some(n) = split.and_then(|(.., v)| v.as_u64()).filter(|n| *n > 1) {
        return refuse(
            Code::UNKNOWN_FORMAT,
            format!("a split GGUF ({n} files): merge it with llama-gguf-split --merge"),
        );
    }
    let mut tensors: Vec<Info> = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut expected = 0u64;
    for _ in 0..n_tensors {
        let name = c.string()?;
        if !names.insert(name.clone()) {
            return refuse(Code::DUPLICATE_KEY, format!("GGUF tensor {name} twice"));
        }
        let dims = c.u32()?;
        if dims > MAX_DIMS {
            return refuse(Code::RANK_CAP, format!("{name}: {dims} dimensions"));
        }
        let ne = (0..dims).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        let id = c.u32()?;
        let offset = c.u64()?;
        let Some(ty) = Type::by_id(id) else {
            return refuse(Code::DTYPE_UNKNOWN, format!("{name}: ggml type {id}"));
        };
        let numel = ne.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
        let rows_whole = ne.first().is_none_or(|n0| n0.is_multiple_of(ty.block));
        let Some(nbytes) = numel.filter(|_| rows_whole).and_then(|n| ty.bytes(n)) else {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{name}: ne {ne:?} is not whole {} blocks", ty.name),
            );
        };
        if offset != expected {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{name}: offset {offset}, llama.cpp expects {expected}"),
            );
        }
        expected = pad(offset + nbytes, alignment);
        tensors.push(Info {
            name,
            ne,
            ty,
            offset,
            nbytes,
        });
    }
    let data = pad(c.pos, alignment);
    if let Some(last) = tensors.last() {
        if data + last.offset + last.nbytes > c.len {
            return refuse(
                Code::CARRIER_TRUNCATED,
                format!("{}: past the file's {} bytes", last.name, c.len),
            );
        }
    }
    Ok(Gguf {
        version,
        kv,
        tensors,
        alignment,
        data,
    })
}

/// `general.alignment` (a power of two, `u32`), else ggml's default.
pub fn alignment(kv: &[Kv]) -> Result<u64> {
    match kv.iter().find(|(k, ..)| k == "general.alignment") {
        None => Ok(DEFAULT_ALIGNMENT),
        Some((_, kind, v)) => match v.as_u64() {
            Some(a) if kind == "u32" && a.is_power_of_two() => Ok(a),
            _ => refuse(
                Code::CARRIER_GEOMETRY,
                format!("general.alignment {kind} {v}: a u32 power of two"),
            ),
        },
    }
}

/// The GGUF header for `kv` and `tensors` (name, ne, type), padded to the alignment: what
/// llama.cpp and gguf-py write before the data section. Each tensor's offset follows the last,
/// padded to the alignment.
pub fn header(version: u32, kv: &[Kv], tensors: &[(String, Vec<u64>, Type)]) -> Result<Vec<u8>> {
    let alignment = alignment(kv)?;
    let mut out = b"GGUF".to_vec();
    out.extend(version.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for (key, kind, value) in kv {
        string(key, &mut out);
        out.extend(tag(kind)?.to_le_bytes());
        write_value(kind, value, &mut out)?;
    }
    let mut offset = 0u64;
    for (name, ne, ty) in tensors {
        string(name, &mut out);
        out.extend((ne.len() as u32).to_le_bytes());
        ne.iter().for_each(|n| out.extend(n.to_le_bytes()));
        out.extend(ty.id.to_le_bytes());
        out.extend(offset.to_le_bytes());
        let nbytes = ty
            .bytes(ne.iter().product())
            .expect("checked by the caller");
        offset = pad(offset + nbytes, alignment);
    }
    out.resize(pad(out.len() as u64, alignment) as usize, 0);
    Ok(out)
}

// ------------------------------------------------------------------------------ the record

/// `<file>.json`: everything of a GGUF but its tensor bytes, and where each tensor went.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    pub version: u32,
    /// `[key, type, value]`, in the file's order.
    pub kv: Vec<Kv>,
    pub tensors: Vec<Row>,
}

/// One GGUF tensor: its bytes are its sources' concatenated, then rows re-permuted when
/// `permute` names llama's head count.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub ne: Vec<u64>,
    /// `[component, key, role]`; role `""` for a plain tensor.
    pub from: Vec<(String, String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permute: Option<u64>,
}

// ------------------------------------------------------------------------------ name maps

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Arch {
    /// The Transformers class, and the one when the file holds expert stacks.
    class: String,
    #[serde(default)]
    class_experts: Option<String>,
    /// GGUF base name (`blk.{bid}.attn_q`) → upstream name; `{xid}` splits an expert stack.
    tensors: BTreeMap<String, String>,
    /// Tensors only llama.cpp has, kept under their GGUF names.
    #[serde(default)]
    llama_cpp_only: Vec<String>,
    /// Rows llama's converter permuted, and the KV (under the arch) naming their head count.
    #[serde(default)]
    permute: BTreeMap<String, String>,
}

fn arches() -> &'static HashMap<String, Arch> {
    static ARCHES: OnceLock<HashMap<String, Arch>> = OnceLock::new();
    ARCHES.get_or_init(|| {
        serde_json::from_str(include_str!("gguf_names.json")).expect("the name maps parse")
    })
}

/// `blk.12.attn_q` → (`blk.{bid}.attn_q`, `12`).
fn template(base: &str) -> (String, Option<&str>) {
    match base.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
        Some((n, rest)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
            (format!("blk.{{bid}}.{rest}"), Some(n))
        }
        _ => (base.to_owned(), None),
    }
}

// ------------------------------------------------------------------------------ the plan

/// One part's bytes: absolute ranges of the file, concatenated.
#[derive(Debug, Clone)]
pub struct OutPart {
    pub role: &'static str,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub ranges: Vec<(u64, u64)>,
}

#[derive(Debug, Clone)]
pub struct Out {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub encoding: Option<Encoding>,
    pub parts: Vec<OutPart>,
}

#[derive(Debug)]
pub struct Planned {
    pub tensors: Vec<Out>,
    pub meta: Meta,
    /// The arch whose name map renamed the tensors, and its Transformers class.
    pub arch: Option<String>,
    pub class: Option<String>,
    /// Every tensor has an upstream name (or is one only llama.cpp has).
    pub normalized: bool,
    /// The upstream repository the file names as its source (`hf://org/repo`), for an arch with
    /// a name map: its configs and tokenizers come with the model.
    pub reference: Option<String>,
}

fn geometry<T>(what: String) -> Result<T> {
    refuse(Code::CARRIER_GEOMETRY, what)
}

/// The logical shape: ComfyUI-GGUF's original shape when it states one, else `ne` reversed.
fn logical(g: &Gguf, t: &Info) -> Result<Vec<u64>> {
    let reversed: Vec<u64> = t.ne.iter().rev().copied().collect();
    let Some(orig) = g.get(&format!("comfy.gguf.orig_shape.{}", t.name)) else {
        return Ok(reversed);
    };
    let dims: Option<Vec<u64>> = match orig.as_array().map(Vec::as_slice) {
        Some([_, Json::Array(items)]) => items.iter().map(Json::as_u64).collect(),
        _ => None,
    };
    match dims {
        Some(d) if d.iter().product::<u64>() == t.ne.iter().product::<u64>() => Ok(d),
        _ => geometry(format!("{}: original shape {orig}", t.name)),
    }
}

fn head_count(g: &Gguf, arch: &str, kv: &str) -> Result<u64> {
    let key = format!("{arch}.{kv}");
    let found = match g.get(&key) {
        None if kv == "attention.head_count_kv" => g.get(&format!("{arch}.attention.head_count")),
        v => v,
    };
    match found.and_then(Json::as_u64) {
        Some(n) if n > 0 => Ok(n),
        _ => geometry(format!("{key}: a head count to undo the Q/K permutation")),
    }
}

/// Row `r` of `rows` (llama's converter order, `heads` heads) holds upstream row `perm(r)`:
/// upstream `[head, half, i]` is GGUF `[head, i, half]`.
pub(crate) fn permuted(rows: u64, heads: u64, upstream: u64) -> u64 {
    let per = rows / heads;
    let half = per / 2;
    let (a, rest) = (upstream / per, upstream % per);
    let (j, i) = (rest / half, rest % half);
    a * per + 2 * i + j
}

/// Plans `g`'s tensors into `component`: renamed by its arch's name map, split, un-permuted
/// and folded.
pub fn plan(g: &Gguf, component: &str) -> Result<Planned> {
    let arch_name = g.get("general.architecture").and_then(Json::as_str);
    let arch = arch_name.and_then(|a| arches().get(a).map(|m| (a, m)));
    let by_name: HashMap<&str, &Info> = g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    // ggml's nvfp4 keeps ModelOpt's per-tensor scales beside the weight.
    let mut folded: HashMap<&str, Vec<(&'static str, &Info)>> = HashMap::new();
    for t in g.tensors.iter().filter(|t| t.ty.name == "nvfp4") {
        let Some(base) = t.name.strip_suffix(".weight") else {
            continue;
        };
        for role in ["scale", "input_scale"] {
            if let Some(s) = by_name.get(format!("{base}.{role}").as_str()) {
                if s.ty.name != "f32" {
                    return geometry(format!("{}: {} is not f32", t.name, s.name));
                }
                folded.entry(&t.name).or_default().push((role, s));
            }
        }
    }
    let consumed: Vec<&str> = folded
        .values()
        .flatten()
        .map(|(_, s)| s.name.as_str())
        .collect();
    let (mut normalized, mut split) = (true, false);
    let mut tensors: Vec<Out> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut scale_rows: HashMap<&str, Vec<(String, String, String)>> = HashMap::new();
    for t in &g.tensors {
        if consumed.contains(&t.name.as_str()) {
            continue;
        }
        let shape = logical(g, t)?;
        let (base, suffix) = t.name.rsplit_once('.').unwrap_or((&t.name, ""));
        let (tmpl, bid) = template(base);
        let mut key = t.name.clone();
        let mut experts = None;
        let mut heads = None;
        if let Some((a, m)) = arch {
            match m.tensors.get(&tmpl) {
                Some(up) => {
                    let up = up.replace("{bid}", bid.unwrap_or(""));
                    key = match suffix {
                        "" => up.clone(),
                        s => format!("{up}.{s}"),
                    };
                    if up.contains("{xid}") {
                        split = true;
                        match shape.as_slice() {
                            [e, _, _] => experts = Some(*e),
                            _ => {
                                return geometry(format!(
                                    "{}: an expert stack of {shape:?}",
                                    t.name
                                ))
                            }
                        }
                    }
                    if let Some(kv) = m.permute.get(&tmpl) {
                        heads = Some(head_count(g, a, kv)?);
                    }
                }
                None => normalized &= m.llama_cpp_only.contains(&tmpl),
            }
        } else {
            normalized = false;
        }
        let start = g.data + t.offset;
        let encoding = plain(t.ty).is_none().then(|| t.ty.encoding());
        let dtype = plain(t.ty).unwrap_or(Dtype::F32);
        let scales = folded.get(t.name.as_str()).cloned().unwrap_or_default();
        let count = experts.unwrap_or(1);
        let shape = match experts {
            Some(_) => shape[1..].to_vec(),
            None => shape,
        };
        let each = t.nbytes / count;
        let mut from = Vec::new();
        for e in 0..count {
            let key = key.replace("{xid}", &e.to_string());
            let mut ranges = vec![(start + e * each, each)];
            if let Some(h) = heads {
                let n = shape.first().copied().unwrap_or(1);
                if !n.is_multiple_of(2 * h) || h == 0 {
                    return geometry(format!("{}: {n} rows are not {h} heads of pairs", t.name));
                }
                let row = each / n;
                ranges = (0..n)
                    .map(|r| (start + permuted(n, h, r) * row, row))
                    .collect();
            }
            let role = if encoding.is_some() { "blocks" } else { "" };
            let mut parts = vec![OutPart {
                role,
                dtype: encoding.as_ref().map_or(dtype.clone(), |_| Dtype::U8),
                shape: match encoding {
                    Some(_) => blocks_shape(t.ty, &shape).map_or_else(
                        || geometry(format!("{}: {shape:?} is not whole blocks", t.name)),
                        Ok,
                    )?,
                    None => shape.clone(),
                },
                ranges,
            }];
            for (role, s) in &scales {
                let one = match s.ne.iter().product::<u64>() {
                    1 => 0,
                    n if n == count => e,
                    n => return geometry(format!("{}: {n} scales for {count}", s.name)),
                };
                parts.push(OutPart {
                    role,
                    dtype: Dtype::F32,
                    shape: vec![1],
                    ranges: vec![(g.data + s.offset + 4 * one, 4)],
                });
                if one == e {
                    scale_rows.entry(s.name.as_str()).or_default().push((
                        component.to_owned(),
                        key.clone(),
                        (*role).to_owned(),
                    ));
                }
            }
            from.push((component.to_owned(), key.clone(), role.to_owned()));
            tensors.push(Out {
                key,
                dtype: dtype.clone(),
                shape: shape.clone(),
                encoding: encoding.clone(),
                parts,
            });
        }
        rows.push(Row {
            name: t.name.clone(),
            ty: t.ty.name.to_owned(),
            ne: t.ne.clone(),
            from,
            permute: heads,
        });
    }
    // The folded scales keep their place in the file's tensor order.
    let mut ordered = Vec::with_capacity(g.tensors.len());
    let mut rows = rows.into_iter();
    for t in &g.tensors {
        match scale_rows.remove(t.name.as_str()) {
            Some(from) => ordered.push(Row {
                name: t.name.clone(),
                ty: t.ty.name.to_owned(),
                ne: t.ne.clone(),
                from,
                permute: None,
            }),
            None => ordered.push(rows.next().expect("a row per unfolded tensor")),
        }
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(o) = tensors.iter().find(|o| !seen.insert(o.key.as_str())) {
        return refuse(
            Code::DUPLICATE_KEY,
            format!("two GGUF tensors become {}", o.key),
        );
    }
    let reference = arch.and_then(|_| {
        ["general.source.url", "general.source.repo_url"]
            .iter()
            .find_map(|k| g.get(k).and_then(Json::as_str).and_then(huggingface))
    });
    Ok(Planned {
        tensors,
        meta: Meta {
            version: g.version,
            kv: g.kv.clone(),
            tensors: ordered,
        },
        arch: arch.map(|(a, _)| a.to_owned()),
        class: arch.map(|(_, m)| match (&m.class_experts, split) {
            (Some(c), true) => c.clone(),
            _ => m.class.clone(),
        }),
        normalized,
        reference,
    })
}

/// `https://huggingface.co/<org>/<repo>` as `hf://<org>/<repo>`.
fn huggingface(url: &str) -> Option<String> {
    let path = url
        .strip_prefix("https://huggingface.co/")?
        .trim_end_matches('/');
    let (org, repo) = path.split_once('/')?;
    let ok = |s: &str| !s.is_empty() && !s.contains(['/', '@', '?', '#']);
    (ok(org) && ok(repo)).then(|| format!("hf://{org}/{repo}"))
}

/// Reads a part's ranges of `file` in order.
pub struct Ranges<'f> {
    file: &'f File,
    ranges: &'f [(u64, u64)],
    at: usize,
    done: u64,
}

impl<'f> Ranges<'f> {
    pub fn new(file: &'f File, ranges: &'f [(u64, u64)]) -> Ranges<'f> {
        Ranges {
            file,
            ranges,
            at: 0,
            done: 0,
        }
    }
}

impl Read for Ranges<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while let Some((start, len)) = self.ranges.get(self.at) {
            if self.done == *len {
                (self.at, self.done) = (self.at + 1, 0);
                continue;
            }
            let n = buf.len().min((len - self.done) as usize);
            self.file.read_exact_at(&mut buf[..n], start + self.done)?;
            self.done += n as u64;
            return Ok(n);
        }
        Ok(0)
    }
}
