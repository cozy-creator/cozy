//! Quantized carriers. A producer stores one quantized weight as role-suffixed tensors
//! (`.weight`, `.weight_scale`, ...); they fold into one encoded tensor whose parts are the
//! producer's bytes unchanged, their packing stated in params (spec §6 rule 1). The producer is
//! read from its own declaration, and which suffixes, dtypes and shapes make which scheme is
//! reviewed data (`carriers.json`). Nothing is inferred from names: an undeclared, unknown or
//! ambiguous carrier is refused with what was found.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;

use platform::durable::FileAt;
use serde::Deserialize;
use serde_json::Value as Json;

use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{Dtype, Encoding, Value};

use super::source::{Group, SourceSet, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Producer {
    Modelopt,
    CompressedTensors,
    Comfy,
    /// `quant_method: fp8` (DeepSeek's and transformers' FineGrainedFP8: DeepSeek, Qwen, Kimi,
    /// MiniMax, GLM).
    Deepseek,
    /// `quant_method: mxfp4`.
    GptOss,
    /// AMD Quark, `quant_method: quark`.
    Quark,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Carrier {
    pub producer: Producer,
    /// The producer's names for the format: ComfyUI's `format`, ModelOpt's `quant_algo`,
    /// compressed-tensors' `format`.
    pub formats: Vec<String>,
    /// ComfyUI's `convrot` flag.
    #[serde(default)]
    pub convrot: bool,
    /// The data part first.
    pub parts: Vec<PartRule>,
    pub encoding: Encoding,
    /// Params a module's marker states: param -> (marker key, the producer's own default).
    #[serde(default)]
    pub marker_params: BTreeMap<String, (String, u64)>,
    /// Params naming a part's stored dtype: param -> part role (`scale_dtype` -> `scale`).
    #[serde(default)]
    pub dtype_params: BTreeMap<String, String>,
    /// Marker keys whose value this carrier decodes, and that value.
    #[serde(default)]
    pub marker_values: BTreeMap<String, Json>,
    /// Params the declaration states: param -> its key in the module's declaration, else the
    /// producer's (`block_size` -> DeepSeek's `weight_block_size`).
    #[serde(default)]
    pub config_params: BTreeMap<String, String>,
    /// Parts are `{module}_{suffix}` and the folded tensor is `{module}` (gpt-oss), not
    /// `{module}.{suffix}` and `{module}.weight`.
    #[serde(default)]
    pub bare: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartRule {
    pub suffix: String,
    pub role: String,
    /// What the producer stores.
    pub dtypes: Vec<Dtype>,
    /// The part's dtype here, when the bytes mean more than the producer's container says.
    #[serde(default, rename = "as")]
    pub stored: Option<Dtype>,
    pub shape: Geometry,
    #[serde(default)]
    pub optional: bool,
    /// The data part's last two stored axes are one row (gpt-oss's `[.., C/32, 16]` blocks).
    #[serde(default)]
    pub merge: bool,
}

/// A part's shape against the logical `[*lead, C]`, with `R = Π lead`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Geometry {
    /// The logical shape: the data part (packed sub-byte elements along the last axis).
    Logical,
    /// One element.
    One,
    /// `R` elements: `[*lead]` or `[*lead, 1]`.
    Rows,
    /// One element, or one per matrix of a stack `[*stack, N, C]`: `[*stack]` (ComfyUI's
    /// `weight_scale_2 [E]` beside `[E, N, C]` experts).
    Matrices,
    /// `C` elements, `[C]` (AWQ's `pre_quant_scale`).
    Columns,
    /// Row-major block scales, `[*lead, ⌈C/n⌉]`.
    Blocks(u64),
    /// Block scales in cuBLAS 128×4 tiles, each matrix of `[*stack, N, C]` on its own:
    /// `[*stack, 128·⌈N/128⌉, 4·⌈W/4⌉]` with `W = ⌈C/n⌉`.
    Tiles(u64),
    /// One scale per tile of the encoding's `block_size` `[r, c]` over each matrix `[N, C]`:
    /// `[*lead[..-1], ⌈N/r⌉, ⌈C/c⌉]`.
    BlockTiles,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    /// Declarations of producers no carrier reads yet: refused, naming the producer.
    pub unread_declarations: Vec<UnreadDeclaration>,
    /// Quantization parts no scheme reads yet: refused wherever they appear.
    pub companions: Vec<Companions>,
    /// ComfyUI marker keys that change how a module runs, never its values.
    pub marker_execution: Vec<String>,
    pub carriers: Vec<Carrier>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnreadDeclaration {
    /// A file beside the weights, or a safetensors metadata key prefix.
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub metadata_prefix: Option<String>,
    pub producer: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Companions {
    pub producer: String,
    /// Key patterns: `*` at either end matches any text.
    pub keys: Vec<String>,
}

/// The reviewed table, parsed once.
pub fn table() -> &'static Table {
    static PARSED: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    PARSED.get_or_init(|| {
        serde_json::from_str(include_str!("carriers.json")).expect("the carrier table")
    })
}

pub fn carriers() -> &'static [Carrier] {
    &table().carriers
}

/// `key` against a pattern with an optional `*` at either end.
fn matches(pattern: &str, key: &str) -> bool {
    match (pattern.strip_prefix('*'), pattern.strip_suffix('*')) {
        (Some(rest), _) if rest.ends_with('*') => key.contains(&rest[..rest.len() - 1]),
        (Some(rest), _) => key.ends_with(rest),
        (None, Some(rest)) => key.starts_with(rest),
        (None, None) => key == pattern,
    }
}

/// One folded tensor: its encoding and each part's role, dtype, shape and source bytes.
#[derive(Debug, Clone)]
pub struct Folded {
    pub encoding: Encoding,
    pub parts: Vec<FoldedPart>,
}

#[derive(Debug, Clone)]
pub struct FoldedPart {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub at: Tensor,
}

/// A module's own declaration: ComfyUI's marker or `_quantization_metadata` layer, ModelOpt's
/// `quantized_layers` entry.
type Conf = serde_json::Map<String, Json>;

/// What a group's producer declared, and where.
struct Declared {
    producer: Producer,
    found: Vec<String>,
    /// Formats any quantized module may have; a module's own `Conf` overrides.
    formats: BTreeSet<String>,
    modules: BTreeMap<String, Conf>,
    /// Keys that only declare (markers), dropped from the output.
    markers: Vec<String>,
    /// The producer's own settings that `config_params` read when a module's declaration does not
    /// state them.
    config: Conf,
    /// The model's dtype, when its `config.json` states it.
    dtype: Option<Dtype>,
}

fn malformed(what: &str, e: impl std::fmt::Display) -> Refusal {
    Refusal {
        code: Code::MALFORMED_JSON,
        detail: format!("{what}: {e}"),
    }
}

/// ComfyUI reads a marker key at its top level, then under `params`.
fn conf_get<'c>(conf: &'c Conf, key: &str) -> Option<&'c Json> {
    conf.get(key)
        .or_else(|| conf.get("params")?.as_object()?.get(key))
}

impl Declared {
    fn new(producer: Producer, found: String) -> Declared {
        Declared {
            producer,
            found: vec![found],
            formats: BTreeSet::new(),
            modules: BTreeMap::new(),
            markers: Vec::new(),
            config: Conf::new(),
            dtype: None,
        }
    }

    /// ComfyUI's declaration, its first find naming it.
    fn comfy(slot: &mut Option<Declared>, found: String) -> &mut Declared {
        slot.get_or_insert_with(|| Declared::new(Producer::Comfy, found))
    }

    fn module(&mut self, module: &str, conf: Conf, what: &str) -> Result<()> {
        match self.modules.get(module) {
            Some(other) if *other != conf => refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!("{module}: {what} declares {conf:?}, another declaration {other:?}"),
            ),
            _ => {
                self.modules.insert(module.to_owned(), conf);
                Ok(())
            }
        }
    }
}

impl Declared {
    /// A producer setting `config_params` may read; two different values for one are refused.
    fn setting(&mut self, key: &str, value: &Json, what: &str) -> Result<()> {
        match self.config.get(key) {
            Some(other) if other != value => refuse(
                Code::AMBIGUOUS_CLASSIFICATION,
                format!("{what}: {key} is both {other} and {value}"),
            ),
            _ => {
                self.config.insert(key.to_owned(), value.clone());
                Ok(())
            }
        }
    }
}

/// AMD Quark's declaration: a global weight spec, exact-name per-module specs
/// (`layer_quant_config`) and `exclude` (modules left unquantized, so carrying no carrier). A spec
/// becomes a format the carrier table names: `mxfp4` (FP4 per 32, E8M0 scales, packed
/// `reorder`, which is §5's low-first order: proven on amd/gpt-oss-20b-w-mxfp4-a-bf16's bytes) or
/// `fp8_block` (E4M3 per tile of `block_size`). Anything else stays an unread format.
fn quark(doc: &Json, what: &str) -> Result<Declared> {
    let mut d = Declared::new(Producer::Quark, what.to_owned());
    let unread = |detail: String| refuse(Code::UNREGISTERED_ENCODING, format!("{what}: {detail}"));
    let export = &doc["export"];
    if export["weight_format"] != "real_quantized" {
        return unread(format!("Quark weight_format {}", export["weight_format"]));
    }
    let pack = export["pack_method"].as_str().unwrap_or_default();
    let format = |w: &Json| -> String {
        let (dtype, scheme) = (&w["dtype"], &w["qscheme"]);
        match (dtype.as_str(), scheme.as_str()) {
            (Some("fp4"), Some("per_group"))
                if w["group_size"] == 32 && w["scale_format"] == "e8m0" && pack == "reorder" =>
            {
                "mxfp4".into()
            }
            (Some("fp8_e4m3"), Some("per_block")) if w["block_size"].is_array() => {
                "fp8_block".into()
            }
            _ => format!("{dtype} {scheme} (pack_method {pack})"),
        }
    };
    let types = doc.get("layer_type_quant_config").and_then(Json::as_object);
    if types.is_some_and(|t| !t.is_empty()) {
        return unread("Quark layer_type_quant_config is not read".into());
    }
    let global = &doc["global_quant_config"]["weight"];
    if !global.is_null() {
        d.formats.insert(format(global));
        if global["block_size"].is_array() {
            d.setting("block_size", &global["block_size"], what)?;
        }
    }
    let layers = doc.get("layer_quant_config").and_then(Json::as_object);
    for (module, spec) in layers.into_iter().flatten() {
        if module.contains(['*', '?', '[']) || module.starts_with("re:") {
            return unread(format!("Quark layer_quant_config pattern {module:?}"));
        }
        let w = &spec["weight"];
        let mut conf = Conf::new();
        conf.insert("format".into(), format(w).into());
        conf.insert("block_size".into(), w["block_size"].clone());
        d.module(module, conf, what)?;
    }
    Ok(d)
}

/// `quantization_config` (Transformers' `config.json` key, or a file's metadata) and ModelOpt's
/// `hf_quant_config.json` `quantization`.
fn quantization_config(doc: &Json, what: &str) -> Result<Declared> {
    let method = doc.get("quant_method").and_then(Json::as_str);
    let mut d = match method {
        Some("modelopt") | None if doc.get("quant_algo").is_some() => {
            let mut d = Declared::new(Producer::Modelopt, what.to_owned());
            let algo = doc["quant_algo"].as_str().unwrap_or_default();
            if algo == "MIXED_PRECISION" {
                let layers = doc.get("quantized_layers").and_then(Json::as_object);
                for (module, conf) in layers.into_iter().flatten() {
                    let mut conf = conf.as_object().cloned().unwrap_or_default();
                    if let Some(algo) = conf.remove("quant_algo") {
                        conf.insert("format".into(), algo);
                    }
                    d.module(module, conf, what)?;
                }
            } else {
                d.formats.insert(algo.to_owned());
            }
            d
        }
        Some("compressed-tensors") => {
            let mut d = Declared::new(Producer::CompressedTensors, what.to_owned());
            let groups = doc.get("config_groups").and_then(Json::as_object);
            let formats = groups
                .into_iter()
                .flat_map(|g| g.values())
                .chain([doc])
                .filter_map(|g| g.get("format")?.as_str());
            d.formats.extend(formats.map(str::to_owned));
            // The block strategy is its own carrier: `float-quantized/block`, its tile from
            // `block_structure`.
            let top = doc.get("format").and_then(Json::as_str);
            for g in groups.into_iter().flat_map(|g| g.values()) {
                let format = g.get("format").and_then(Json::as_str).or(top);
                let w = &g["weights"];
                if format == Some("float-quantized") && w["strategy"] == "block" {
                    d.formats.insert("float-quantized/block".into());
                    d.setting("block_structure", &w["block_structure"], what)?;
                }
            }
            d
        }
        Some("fp8") => {
            let mut d = Declared::new(Producer::Deepseek, what.to_owned());
            if !matches!(doc.get("fmt").and_then(Json::as_str), None | Some("e4m3")) {
                return refuse(
                    Code::UNREGISTERED_ENCODING,
                    format!("{what}: quant_method fp8 with fmt {}", doc["fmt"]),
                );
            }
            match doc.get("weight_block_size") {
                Some(b @ Json::Array(_)) => d.setting("weight_block_size", b, what)?,
                _ => {
                    return refuse(
                        Code::UNREGISTERED_ENCODING,
                        format!("{what}: quant_method fp8 with no weight_block_size (per-tensor)"),
                    )
                }
            }
            d.formats.insert("fp8".into());
            d
        }
        Some("mxfp4") => {
            let mut d = Declared::new(Producer::GptOss, what.to_owned());
            d.formats.insert("mxfp4".into());
            d
        }
        Some("quark") => quark(doc, what)?,
        other => {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{what} declares quantization {}, which this import does not read",
                    other.map_or_else(|| doc.to_string(), |m| format!("quant_method {m:?}"))
                ),
            )
        }
    };
    d.formats.remove("mixed-precision");
    Ok(d)
}

fn read_json(set: &SourceSet, path: &str) -> Result<Option<Json>> {
    let Some(plain) = set.plain.iter().find(|p| p.path == path) else {
        return Ok(None);
    };
    let bytes = set.read_plain(plain)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| malformed(path, e))
}

fn bytes(set: &SourceSet, key: &str, t: &Tensor) -> Result<Vec<u8>> {
    if t.len > 1 << 20 {
        return refuse(Code::SIZE_CAP, format!("{key}: a {} byte marker", t.len));
    }
    let path = &set.files[t.file];
    let io = |e| crate::err::classify_io(format!("read {key} in {}", path.display()), e);
    let mut buf = vec![0; t.len as usize];
    File::open(path)
        .and_then(|f| f.read_exact_at(&mut buf, t.start))
        .map_err(io)?;
    Ok(buf)
}

/// Every declaration of `group`'s producer: refused when two producers declare it.
fn declared(set: &SourceSet, group: &Group) -> Result<Option<Declared>> {
    let mut all: Vec<Declared> = Vec::new();
    let path = |name: &str| match group.folder.as_str() {
        "" => name.to_owned(),
        folder => format!("{folder}/{name}"),
    };
    for unread in &table().unread_declarations {
        let found = match (&unread.file, &unread.metadata_prefix) {
            (Some(file), _) => read_json(set, &path(file))?.map(|doc| {
                let stated: Vec<String> = ["quant_method", "checkpoint_format", "version"]
                    .iter()
                    .filter_map(|k| Some(format!("{k}: {}", doc.get(*k)?)))
                    .collect();
                format!("{} ({})", path(file), stated.join(", "))
            }),
            (None, Some(prefix)) => (group.metadata.keys())
                .find(|k| k.starts_with(prefix.as_str()))
                .map(|k| format!("{} metadata {k}", group.label())),
            (None, None) => None,
        };
        if let Some(found) = found {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{found}: a {} quantization, which this import does not read yet",
                    unread.producer
                ),
            );
        }
    }
    if let Some(doc) = read_json(set, &path("hf_quant_config.json"))? {
        let what = path("hf_quant_config.json");
        match doc["producer"]["name"].as_str() {
            Some("modelopt") => all.push(quantization_config(&doc["quantization"], &what)?),
            other => {
                return refuse(
                    Code::UNREGISTERED_ENCODING,
                    format!("{what}: producer {other:?}, which this import does not read"),
                )
            }
        }
    }
    let mut dtype = None;
    if let Some(doc) = read_json(set, &path("config.json"))? {
        if let Some(q) = doc.get("quantization_config") {
            let mut d = quantization_config(q, &path("config.json"))?;
            // DeepSeek-V4's routed experts are FP4 (MXFP4) beside its FP8 tiles.
            if d.producer == Producer::Deepseek && doc["expert_dtype"] == "fp4" {
                d.formats.insert("fp4".into());
            }
            all.push(d);
        }
        let named = doc.get("torch_dtype").or_else(|| doc.get("dtype"));
        dtype = match named.and_then(Json::as_str) {
            Some("bfloat16") => Some(Dtype::BF16),
            Some("float16") => Some(Dtype::F16),
            Some("float32") => Some(Dtype::F32),
            _ => None,
        };
    }
    let label = group.label();
    if let Some(text) = group.metadata.get("quantization_config") {
        let what = format!("{label} metadata quantization_config");
        let doc: Json = serde_json::from_str(text).map_err(|e| malformed(&what, e))?;
        all.push(quantization_config(&doc, &what)?);
    }
    let mut comfy: Option<Declared> = None;
    if let Some(text) = group.metadata.get("_quantization_metadata") {
        let what = format!("{label} metadata _quantization_metadata");
        let doc: Json = serde_json::from_str(text).map_err(|e| malformed(&what, e))?;
        let d = Declared::comfy(&mut comfy, what.clone());
        let layers = doc.get("layers").and_then(Json::as_object);
        for (module, conf) in layers.into_iter().flatten() {
            let conf = conf.as_object().cloned().unwrap_or_default();
            d.module(module, conf, &what)?;
        }
    }
    for (key, t) in &group.tensors {
        if let Some(module) = key.strip_suffix(".comfy_quant") {
            let conf: Conf =
                serde_json::from_slice(&bytes(set, key, t)?).map_err(|e| malformed(key, e))?;
            let d = Declared::comfy(&mut comfy, format!("{label} .comfy_quant markers"));
            d.module(module, conf, key)?;
            d.markers.push(key.clone());
        } else if key == "scaled_fp8" || key.ends_with(".scaled_fp8") {
            // ComfyUI's legacy declaration: every `.scale_weight` module is scaled fp8.
            let d = Declared::comfy(&mut comfy, format!("{label} {key}"));
            d.formats.insert("scaled_fp8".into());
            d.markers.push(key.clone());
        }
    }
    all.extend(comfy);
    let producers: BTreeSet<Producer> = all.iter().map(|d| d.producer).collect();
    if producers.len() > 1 {
        let found: Vec<String> = all.iter().flat_map(|d| d.found.clone()).collect();
        return refuse(
            Code::AMBIGUOUS_CLASSIFICATION,
            format!(
                "{label}: quantization declared by {producers:?} ({}); one producer's carrier \
                 is read, never a guess between two",
                found.join(", ")
            ),
        );
    }
    let mut merged: Option<Declared> = None;
    for d in all {
        match &mut merged {
            None => merged = Some(d),
            Some(m) => {
                m.found.extend(d.found);
                m.formats.extend(d.formats);
                m.markers.extend(d.markers);
                for (key, value) in &d.config {
                    m.setting(key, value, "a second declaration")?;
                }
                for (module, conf) in d.modules {
                    m.module(&module, conf, "a second declaration")?;
                }
            }
        }
    }
    Ok(merged.map(|m| Declared { dtype, ..m }))
}

/// The dtype `group`'s quantized weights stand for: the model's, as its `config.json` states it,
/// else that of its unquantized float tensors (the most common one).
fn logical_dtype(group: &Group, d: &Declared, consumed: &BTreeSet<String>) -> Result<Dtype> {
    if let Some(dtype) = &d.dtype {
        return Ok(dtype.clone());
    }
    let mut counts: BTreeMap<Dtype, usize> = BTreeMap::new();
    for (key, t) in &group.tensors {
        if !consumed.contains(key) && matches!(t.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
            *counts.entry(t.dtype.clone()).or_default() += 1;
        }
    }
    let mut ranked: Vec<(usize, Dtype)> = counts.into_iter().map(|(d, n)| (n, d)).collect();
    ranked.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    match ranked.as_slice() {
        [(_, d)] => Ok(d.clone()),
        [(a, d), (b, _), ..] if a > b => Ok(d.clone()),
        _ => refuse(
            Code::DTYPE_UNKNOWN,
            format!(
                "{}: quantized weights, and neither a config.json dtype nor a single dtype \
                 among its unquantized float tensors ({ranked:?}) to read their logical dtype from",
                group.label()
            ),
        ),
    }
}

fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// The stored shape of `t` as `rule` reads it against the logical `shape`, or why not.
fn geometry(
    rule: &PartRule,
    t: &Tensor,
    shape: &[u64],
    block: Option<[u64; 2]>,
) -> std::result::Result<Vec<u64>, String> {
    let (lead, c) = match shape {
        [lead @ .., c] => (lead, *c),
        [] => (&[][..], 1),
    };
    let blocks = |n: u64| c.div_ceil(n);
    let block_scheme = !matches!(rule.shape, Geometry::Logical | Geometry::One);
    if block_scheme && shape.len() < 2 {
        return Err(format!("a logical shape {shape:?} of rank under 2"));
    }
    let stored = match rule.shape {
        Geometry::Logical => shape.to_vec(),
        Geometry::One => match numel(&t.shape) {
            1 => t.shape.clone(),
            _ => return Err(format!("{:?}, not one element", t.shape)),
        },
        Geometry::Rows => {
            let ok = t.shape == lead || t.shape == [lead, &[1]].concat();
            match ok {
                true => t.shape.clone(),
                false => return Err(format!("{:?}, not {lead:?} or {lead:?}+[1]", t.shape)),
            }
        }
        Geometry::Blocks(n) => {
            let want = [lead, &[blocks(n)]].concat();
            match t.shape == want {
                true => want,
                false => return Err(format!("{:?}, not {want:?}", t.shape)),
            }
        }
        Geometry::Matrices => {
            let stack = &lead[..lead.len() - 1];
            match numel(&t.shape) == 1 || (!stack.is_empty() && t.shape == stack) {
                true => t.shape.clone(),
                false => return Err(format!("{:?}, not one element or {stack:?}", t.shape)),
            }
        }
        Geometry::Columns => match t.shape == [c] {
            true => t.shape.clone(),
            false => return Err(format!("{:?}, not [{c}]", t.shape)),
        },
        Geometry::Tiles(n) => {
            let (stack, rows) = lead.split_at(lead.len() - 1);
            let want = [stack, &[rows[0].div_ceil(128) * 128, blocks(n).div_ceil(4) * 4]].concat();
            match t.shape == want {
                true => want,
                false => return Err(format!("{:?}, not cuBLAS tiles {want:?}", t.shape)),
            }
        }
        Geometry::BlockTiles => {
            let ([outer @ .., n], Some([r, cols])) = (lead, block) else {
                return Err("tiles with no block_size".into());
            };
            let want = [outer, &[n.div_ceil(r), c.div_ceil(cols)]].concat();
            match t.shape == want {
                true => want,
                false => return Err(format!("{:?}, not tiles {want:?}", t.shape)),
            }
        }
    };
    let dtype = rule.stored.as_ref().unwrap_or(&t.dtype);
    match dtype.byte_len(&stored) {
        Some(n) if n == t.len => Ok(stored),
        _ => Err(format!("{} bytes are not {dtype} {stored:?}", t.len)),
    }
}

/// `carrier` read from the module's present tensors (suffix -> tensor), or why not.
fn fit(
    carrier: &Carrier,
    present: &Present,
    conf: Option<&Conf>,
    config: &Conf,
) -> std::result::Result<Folded, String> {
    if let Some(s) = (present.keys()).find(|s| !carrier.parts.iter().any(|r| r.suffix == **s)) {
        return Err(format!("no part for .{s}"));
    }
    let data = &carrier.parts[0];
    let Some((_, t)) = present.get(data.suffix.as_str()) else {
        return Err(format!("no .{}", data.suffix));
    };
    let stored = data.stored.as_ref().unwrap_or(&t.dtype);
    let (Some(from), Some(to)) = (t.dtype.bits(), stored.bits()) else {
        return Err(format!(".{} is {}", data.suffix, t.dtype));
    };
    let mut shape = t.shape.clone();
    if data.merge && shape.len() >= 2 {
        let last = shape.pop().unwrap_or_default();
        if let Some(n) = shape.last_mut() {
            *n *= last;
        }
    }
    if let Some(last) = shape.last_mut() {
        *last = *last * from / to;
    }
    let mut encoding = carrier.encoding.clone();
    for (param, key) in &carrier.config_params {
        let stated = conf.and_then(|c| conf_get(c, key)).filter(|v| !v.is_null());
        let v = stated
            .or_else(|| config.get(key))
            .ok_or(format!("no {key} declares {param}"))?;
        let value = json_value(v).ok_or(format!("{key} is {v}"))?;
        encoding.params.insert(param.clone(), value);
    }
    let block = match encoding.params.get("block_size") {
        Some(Value::List(l)) => match l.iter().map(Value::as_u64).collect::<Vec<_>>()[..] {
            [Some(r), Some(c)] if r > 0 && c > 0 => Some([r, c]),
            _ => return Err(format!("block_size {l:?} is not two positive extents")),
        },
        _ => None,
    };
    let mut parts = Vec::new();
    for rule in &carrier.parts {
        let Some((_, t)) = present.get(rule.suffix.as_str()) else {
            if rule.optional {
                continue;
            }
            return Err(format!("no .{}", rule.suffix));
        };
        if !rule.dtypes.contains(&t.dtype) {
            return Err(format!(
                ".{} is {}, not {:?}",
                rule.suffix, t.dtype, rule.dtypes
            ));
        }
        let part_shape =
            geometry(rule, t, &shape, block).map_err(|e| format!(".{}: {e}", rule.suffix))?;
        parts.push(FoldedPart {
            role: rule.role.clone(),
            dtype: rule.stored.clone().unwrap_or_else(|| t.dtype.clone()),
            shape: part_shape,
            at: (*t).clone(),
        });
    }
    if let (Producer::Comfy, Some(conf)) = (carrier.producer, conf) {
        marker_keys(carrier, conf)?;
    }
    for (param, (key, default)) in &carrier.marker_params {
        let value = match conf.and_then(|c| conf_get(c, key)) {
            None => *default,
            Some(v) => v
                .as_u64()
                .ok_or(format!("marker {key} is {v}, not a count"))?,
        };
        encoding.params.insert(param.clone(), Value::from(value));
    }
    // A param naming an optional part's dtype is there exactly when the part is.
    for (param, role) in &carrier.dtype_params {
        match parts.iter().find(|p| p.role == *role) {
            Some(part) => encoding.params.insert(param.clone(), Value::from(part.dtype.name())),
            None if carrier.parts.iter().any(|r| r.role == *role && r.optional) => continue,
            None => return Err(format!("no {role} part for {param}")),
        };
    }
    Ok(Folded { encoding, parts })
}

/// Every key of a ComfyUI marker (and of its `params`) is one this carrier reads, one it
/// decodes at the stated value, or one that never changes values; any other is refused by name.
fn marker_keys(carrier: &Carrier, conf: &Conf) -> std::result::Result<(), String> {
    let nested = match conf.get("params") {
        None => None,
        Some(Json::Object(params)) => Some(params),
        Some(other) => return Err(format!("marker params: {other} is not a map")),
    };
    let entries = conf
        .iter()
        .filter(|(k, _)| *k != "params")
        .chain(nested.into_iter().flatten());
    for (key, value) in entries {
        let read = key == "format"
            || (key == "convrot" && value.is_boolean())
            || carrier.marker_params.values().any(|(k, _)| k == key)
            || table().marker_execution.contains(key);
        let decoded = carrier.marker_values.get(key).map(|want| want == value);
        match (read, decoded) {
            (true, _) | (false, Some(true)) => {}
            (false, Some(false)) => {
                return Err(format!(
                    "marker {key}: {value}, where {} decodes {}",
                    carrier.encoding.scheme, carrier.marker_values[key]
                ))
            }
            (false, None) => return Err(format!("marker {key}: {value} is not read")),
        }
    }
    Ok(())
}

/// Refuses quantization parts no scheme reads yet, wherever they are.
fn companions(group: &Group) -> Result<()> {
    for c in &table().companions {
        let found: Vec<&String> = (group.tensors.keys())
            .filter(|k| c.keys.iter().any(|p| matches(p, k)))
            .take(3)
            .collect();
        if !found.is_empty() {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{}: {found:?} are {} quantization parts, which this import does not read yet",
                    group.label(),
                    c.producer
                ),
            );
        }
    }
    Ok(())
}

/// A declaration's JSON value as a param: a count, a name, a flag or a list of them.
fn json_value(v: &Json) -> Option<Value> {
    match v {
        Json::Number(n) => n.as_u64().map(Value::from),
        Json::String(s) => Some(Value::from(s.as_str())),
        Json::Bool(b) => Some(Value::from(*b)),
        Json::Array(a) => a
            .iter()
            .map(json_value)
            .collect::<Option<_>>()
            .map(Value::List),
        _ => None,
    }
}

/// How a producer joins a module and a part's suffix: `.` (`{module}.weight_scale`), or `_` for
/// `bare` rows (gpt-oss's `{module}_scales`).
fn joiner(c: &Carrier) -> char {
    if c.bare {
        '_'
    } else {
        '.'
    }
}

/// Folds every quantized carrier in `set` into encoded tensors under the module's `.weight` (or,
/// for `bare` rows, the module's own name).
pub fn fold(set: &mut SourceSet) -> Result<()> {
    for g in 0..set.groups.len() {
        let group = &set.groups[g];
        let Some(d) = declared(set, group)? else {
            undeclared(group)?;
            companions(group)?;
            continue;
        };
        let folded = fold_group(group, &d)?;
        let group = &mut set.groups[g];
        for key in &d.markers {
            group.tensors.remove(key);
        }
        for (key, (consumed, t)) in folded {
            for c in consumed {
                group.tensors.remove(&c);
            }
            group.tensors.insert(key, t);
        }
        companions(group)?;
    }
    Ok(())
}

/// Refuses carrier parts no declaration explains: a suffix of the table beside its row's data
/// part, when that data part is not a plain float tensor.
fn undeclared(group: &Group) -> Result<()> {
    let quantized = |key: &str| {
        (group.tensors.get(key))
            .is_some_and(|t| !matches!(t.dtype, Dtype::F32 | Dtype::BF16 | Dtype::F16))
    };
    let found: Vec<&String> = (group.tensors.keys())
        .filter(|k| {
            carriers().iter().any(|c| {
                let (j, data) = (joiner(c), &c.parts[0].suffix);
                c.parts.iter().skip(1).any(|r| {
                    let module = k
                        .strip_suffix(r.suffix.as_str())
                        .and_then(|m| m.strip_suffix(j));
                    module.is_some_and(|m| quantized(&format!("{m}{j}{data}")))
                })
            })
        })
        .take(4)
        .collect();
    match found.as_slice() {
        [] => Ok(()),
        _ => refuse(
            Code::UNREGISTERED_ENCODING,
            format!(
                "{}: quantized carrier tensors ({found:?}) and no declaration of their producer \
                 (hf_quant_config.json, quantization_config, .comfy_quant, \
                 _quantization_metadata or scaled_fp8)",
                group.label()
            ),
        ),
    }
}

type Folds = BTreeMap<String, (Vec<String>, Tensor)>;

/// A module's carrier tensors present: suffix -> (key, tensor).
type Present<'g> = BTreeMap<&'g str, (&'g str, &'g Tensor)>;

fn fold_group(group: &Group, d: &Declared) -> Result<Folds> {
    let label = group.label();
    let rows: Vec<&Carrier> = carriers()
        .iter()
        .filter(|c| c.producer == d.producer)
        .collect();
    let suffixes: BTreeSet<&str> = rows
        .iter()
        .flat_map(|c| &c.parts)
        .map(|r| r.suffix.as_str())
        .collect();
    let data: BTreeSet<&str> = rows.iter().map(|c| c.parts[0].suffix.as_str()).collect();
    // One producer's rows share a joiner (and so `bare`).
    let bare = rows.iter().any(|c| c.bare);
    let j = if bare { '_' } else { '.' };
    // Modules: each carrier role key's, and each module a declaration names.
    let mut modules: BTreeSet<&str> = BTreeSet::new();
    for key in group.tensors.keys() {
        if let Some((module, s)) = key.rsplit_once(j) {
            if suffixes.contains(s) && !data.contains(s) {
                modules.insert(module);
            }
        }
    }
    for module in d.modules.keys() {
        if data
            .iter()
            .any(|s| group.tensors.contains_key(&format!("{module}{j}{s}")))
        {
            modules.insert(module);
        }
    }
    let present = |module: &str| -> Present {
        (suffixes.iter())
            .filter_map(|s| {
                let (key, t) = group.tensors.get_key_value(&format!("{module}{j}{s}"))?;
                Some((*s, (key.as_str(), t)))
            })
            .collect()
    };
    let mut folds = Vec::new();
    for module in modules {
        let present = present(module);
        let conf = d.modules.get(module);
        let formats: BTreeSet<String> = match conf {
            Some(conf) => match conf.get("format").and_then(Json::as_str) {
                Some(f) => [f.to_owned()].into(),
                None => {
                    return refuse(
                        Code::UNREGISTERED_ENCODING,
                        format!("{label}: {module} is declared with no format: {conf:?}"),
                    )
                }
            },
            None if d.producer == Producer::Comfy && !d.formats.contains("scaled_fp8") => {
                return refuse(
                    Code::UNREGISTERED_ENCODING,
                    format!(
                        "{label}: {module} carries {:?} with no .comfy_quant marker or \
                         _quantization_metadata layer",
                        present.keys().collect::<Vec<_>>()
                    ),
                )
            }
            None => d.formats.clone(),
        };
        let convrot = conf
            .and_then(|c| conf_get(c, "convrot"))
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let candidates: Vec<&&Carrier> = rows
            .iter()
            .filter(|c| c.convrot == convrot && c.formats.iter().any(|f| formats.contains(f)))
            .collect();
        let found = || {
            let found: Vec<String> = (present.values())
                .map(|(k, t)| format!("{k} {} {:?}", t.dtype, t.shape))
                .collect();
            found.join(", ")
        };
        if candidates.is_empty() {
            let known: BTreeSet<&String> = rows.iter().flat_map(|c| &c.formats).collect();
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{label}: {module} is {:?} format {formats:?}{}, which this import does not \
                     read (it reads {known:?}); found {}",
                    d.producer,
                    if convrot { " with convrot" } else { "" },
                    found()
                ),
            );
        }
        let mut fits = Vec::new();
        let mut why = Vec::new();
        for c in &candidates {
            match fit(c, &present, conf, &d.config) {
                Ok(f) => fits.push(f),
                Err(e) => why.push(format!("{}: {e}", c.encoding.scheme)),
            }
        }
        let folded = match fits.len() {
            1 => fits.remove(0),
            0 => {
                return refuse(
                    Code::MISSING_COMPANION_ROLE,
                    format!(
                        "{label}: {module} ({:?} {formats:?}) is no carrier this import reads: \
                         found {}; tried {}",
                        d.producer,
                        found(),
                        why.join("; ")
                    ),
                )
            }
            _ => {
                return refuse(
                    Code::AMBIGUOUS_CLASSIFICATION,
                    format!(
                        "{label}: {module} fits {} carriers: {}",
                        fits.len(),
                        found()
                    ),
                )
            }
        };
        let key = match bare {
            true => module.to_owned(),
            false => format!("{module}.weight"),
        };
        let consumed: Vec<String> = present.values().map(|(k, _)| k.to_string()).collect();
        // A sibling no carrier reads (AWQ's `pre_quant_scale`, a zero point, low-rank terms)
        // changes the weight's value: never left beside it as a plain tensor.
        let prefix = format!("{module}{j}");
        let unread: Vec<&String> = (group.tensors.range(prefix.clone()..))
            .map(|(k, _)| k)
            .take_while(|k| k.starts_with(&prefix))
            .filter(|k| {
                let rest = &k[prefix.len()..];
                !rest.contains('.')
                    && rest != "bias"
                    && !consumed.contains(k)
                    && !d.markers.contains(k)
            })
            .collect();
        if !unread.is_empty() {
            return refuse(
                Code::UNREGISTERED_ENCODING,
                format!(
                    "{label}: {unread:?} beside quantized {key}: no {:?} carrier this import reads \
                     has that part",
                    d.producer
                ),
            );
        }
        if group.tensors.contains_key(&key) && !consumed.contains(&key) {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("{label}: {key} and its quantized carrier both exist"),
            );
        }
        folds.push((key, consumed, folded));
    }
    if folds.is_empty() {
        return Ok(Folds::new());
    }
    let consumed: BTreeSet<String> = folds.iter().flat_map(|(_, c, _)| c.clone()).collect();
    let logical = logical_dtype(group, d, &consumed)?;
    let mut out = Folds::new();
    for (key, consumed, folded) in folds {
        let data = &folded.parts[0];
        // `transposed: true`: the parts are the logical tensor with its last two axes swapped.
        let mut shape = data.shape.clone();
        if folded.encoding.params.get("transposed") == Some(&Value::Bool(true)) {
            let n = shape.len();
            shape.swap(n - 2, n - 1);
        }
        let tensor = Tensor {
            dtype: logical.clone(),
            shape,
            file: data.at.file,
            start: data.at.start,
            len: data.at.len,
            encoded: Some(Box::new(folded)),
        };
        out.insert(key, (consumed, tensor));
    }
    Ok(out)
}
