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
use crate::format::{check, Dtype, Encoding, PartBytes, Value};

use super::source::{Group, SourceSet, Tensor};

mod bnb;
mod nunchaku;

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
    Gptq,
    Awq,
    /// bitsandbytes: `quant_method: bitsandbytes`, or its own `quant_state` documents.
    Bitsandbytes,
    /// Nunchaku's SVDQuant: `quantization_config` `method: svdquant`.
    Nunchaku,
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
    /// ComfyUI always rotates this format: its marker's `convrot` may be absent or true, never false.
    #[serde(default)]
    pub rotated: bool,
    /// The data part first.
    pub parts: Vec<PartRule>,
    pub encoding: Encoding,
    /// Params a module's marker states: param -> (marker key, the producer's own default).
    #[serde(default)]
    pub marker_params: BTreeMap<String, (String, u64)>,
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
    /// Where the logical shape comes from, when the data part does not state it.
    #[serde(default)]
    pub logical: Logical,
    /// The code width (`{bits}` in the scheme) and every param are the producer's quantization
    /// config's (`config_encoding`); the scheme's own rules check the parts.
    #[serde(default)]
    pub from_config: bool,
}

/// A carrier's logical shape.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Logical {
    /// The data part's shape, its last extent counted in the stored dtype's elements.
    #[default]
    Data,
    /// `[N, K]`: an extent of one part's tensor and an extent of another's, by suffix.
    Extents {
        rows: (String, usize),
        cols: (String, usize),
    },
    /// The integers a part holds (compressed-tensors' `weight_shape`).
    Stated(String),
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
    /// As the producer stores it, checked by the scheme's own rules (`from_config`).
    Scheme,
    /// The data part, `n` bits per element along the last axis: `[*lead, C·n/8]`.
    Packed(u64),
    /// Row-major scales per group of the encoding's `group_size`, `[*lead, C/g]`.
    Groups,
    /// The same, transposed: `[C/g, R]`.
    GroupsTransposed,
    /// `n` elements, any shape.
    Count(u64),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    /// Declarations beside the weights that name their producer.
    pub declarations: Vec<Declaration>,
    /// Quantization parts no scheme reads yet: refused wherever they appear.
    pub companions: Vec<Companions>,
    /// ComfyUI marker keys that change how a module runs, never its values.
    pub marker_execution: Vec<String>,
    pub carriers: Vec<Carrier>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    /// A file beside the weights (its own `quant_method`, if it states one, names the producer),
    /// or a safetensors metadata key prefix (the keys without it are the config).
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub metadata_prefix: Option<String>,
    pub producer: Producer,
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
    /// The logical shape (before a `transposed` encoding's swap).
    pub shape: Vec<u64>,
    pub parts: Vec<FoldedPart>,
}

#[derive(Debug, Clone)]
pub struct FoldedPart {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub at: Tensor,
    /// The part's bytes when the producer keeps them in a document, not a tensor (bnb's
    /// `nested_offset`, a JSON float).
    pub inline: Option<Vec<u8>>,
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
    /// The producer's quantization configs, each with where it was found.
    configs: Vec<(String, Json)>,
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
            configs: Vec::new(),
        }
    }

    /// A producer's declaration by its own config alone (`quantize_config.json`, AutoAWQ's
    /// `quant_config.json`, GPTQ's header metadata).
    fn of_config(producer: Producer, what: &str, doc: Json) -> Declared {
        let mut d = Declared::new(producer, what.to_owned());
        d.formats.insert(format_of(producer).to_owned());
        d.configs.push((what.to_owned(), doc));
        d
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
        Some(m @ ("gptq" | "awq")) => {
            let producer = if m == "gptq" {
                Producer::Gptq
            } else {
                Producer::Awq
            };
            Declared::of_config(producer, what, doc.clone())
        }
        Some("bitsandbytes") => Declared::new(Producer::Bitsandbytes, what.to_owned()),
        Some("compressed-tensors") => {
            let mut d = Declared::new(Producer::CompressedTensors, what.to_owned());
            d.configs.push((what.to_owned(), doc.clone()));
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
        None if doc.get("method").and_then(Json::as_str) == Some("svdquant") => {
            let mut d = Declared::new(Producer::Nunchaku, what.to_owned());
            d.configs.push((what.to_owned(), doc.clone()));
            d
        }
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
    for decl in &table().declarations {
        match (&decl.file, &decl.metadata_prefix) {
            (Some(file), _) => {
                if let Some(doc) = read_json(set, &path(file))? {
                    let what = path(file);
                    all.push(match doc.get("quant_method") {
                        Some(_) => quantization_config(&doc, &what)?,
                        None => Declared::of_config(decl.producer, &what, doc),
                    });
                }
            }
            (None, Some(prefix)) => {
                let stated: serde_json::Map<String, Json> = (group.metadata.iter())
                    .filter_map(|(k, v)| {
                        Some((
                            k.strip_prefix(prefix.as_str())?.to_owned(),
                            metadata_value(v),
                        ))
                    })
                    .collect();
                if !stated.is_empty() {
                    let what = format!("{} metadata {prefix}*", group.label());
                    all.push(Declared::of_config(
                        decl.producer,
                        &what,
                        Json::Object(stated),
                    ));
                }
            }
            (None, None) => {}
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
    if let Some(found) = bnb::declaration(group) {
        all.push(Declared::new(Producer::Bitsandbytes, found));
    }
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
                m.configs.extend(d.configs);
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
    group: Option<u64>,
) -> std::result::Result<Vec<u64>, String> {
    let (lead, c) = match shape {
        [lead @ .., c] => (lead, *c),
        [] => (&[][..], 1),
    };
    let blocks = |n: u64| c.div_ceil(n);
    let block_scheme = !matches!(
        rule.shape,
        Geometry::Logical | Geometry::One | Geometry::Scheme | Geometry::Count(_)
    );
    if block_scheme && shape.len() < 2 {
        return Err(format!("a logical shape {shape:?} of rank under 2"));
    }
    let stored = match rule.shape {
        Geometry::Scheme => t.shape.clone(),
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
        Geometry::Packed(bits) => {
            let width = match bits {
                4 if c.is_multiple_of(2) => Some(c / 2),
                6 if c.is_multiple_of(4) => (c / 4).checked_mul(3),
                _ => None,
            }
            .ok_or_else(|| format!("{c} columns are not whole {bits}-bit rows"))?;
            [lead, &[width]].concat()
        }
        Geometry::Count(n) => match numel(&t.shape) == n {
            true => t.shape.clone(),
            false => return Err(format!("{:?}, not {n} elements", t.shape)),
        },
        Geometry::Groups | Geometry::GroupsTransposed => {
            let g = group.filter(|g| *g > 0 && c % g == 0);
            let Some(g) = g else {
                return Err(format!("{c} columns are not whole groups of {group:?}"));
            };
            let want = match rule.shape {
                Geometry::Groups => [lead, &[c / g]].concat(),
                _ => vec![c / g, numel(lead)],
            };
            match t.shape == want {
                true => want,
                false => return Err(format!("{:?}, not {want:?}", t.shape)),
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
    configured: Option<&Configured>,
    read: &dyn Fn(&str, &Tensor) -> Result<Vec<u8>>,
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
    let shape = match &carrier.logical {
        Logical::Data => {
            let mut shape = t.shape.clone();
            if data.merge && shape.len() >= 2 {
                let last = shape.pop().unwrap_or_default();
                if let Some(n) = shape.last_mut() {
                    *n *= last;
                }
            }
            if let Some(last) = shape.last_mut() {
                *last = match data.shape {
                    Geometry::Packed(bits) => {
                        let expanded = match bits {
                            4 => last.checked_mul(2),
                            6 if last.is_multiple_of(3) => (*last / 3).checked_mul(4),
                            _ => return Err(format!(".{}: not {bits}-bit rows", data.suffix)),
                        };
                        expanded.ok_or_else(|| {
                            format!(".{}: packed logical width overflow", data.suffix)
                        })?
                    }
                    _ => *last * from / to,
                };
            }
            shape
        }
        logical => logical_shape(logical, present, read)?,
    };
    let mut encoding = carrier.encoding.clone();
    for (param, (key, default)) in &carrier.marker_params {
        let value = match conf.and_then(|c| conf_get(c, key)) {
            None => *default,
            Some(v) => v
                .as_u64()
                .ok_or(format!("marker {key} is {v}, not a count"))?,
        };
        encoding.params.insert(param.clone(), Value::from(value));
    }
    let group = encoding.params.get("group_size").and_then(Value::as_u64);
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
        let part_shape = geometry(rule, t, &shape, block, group)
            .map_err(|e| format!(".{}: {e}", rule.suffix))?;
        parts.push(FoldedPart {
            role: rule.role.clone(),
            dtype: rule.stored.clone().unwrap_or_else(|| t.dtype.clone()),
            shape: part_shape,
            at: (*t).clone(),
            inline: None,
        });
    }
    if carrier.from_config {
        let configured = configured.ok_or("no quantization config states its params")?;
        let encoding = config_encoding(carrier, configured, &shape, &parts)?;
        return Ok(Folded {
            encoding,
            shape,
            parts,
        });
    }
    if let (Producer::Comfy, Some(conf)) = (carrier.producer, conf) {
        marker_keys(carrier, conf)?;
    }
    Ok(Folded {
        encoding,
        shape,
        parts,
    })
}

/// A logical shape the data part does not state: two parts' extents, or a part's integers.
fn logical_shape(
    logical: &Logical,
    present: &Present,
    read: &dyn Fn(&str, &Tensor) -> Result<Vec<u8>>,
) -> std::result::Result<Vec<u64>, String> {
    match logical {
        Logical::Data => unreachable!("the data part's shape"),
        Logical::Extents { rows, cols } => {
            let extent = |(suffix, axis): &(String, usize)| {
                let t = present.get(suffix.as_str()).map(|(_, t)| t);
                t.and_then(|t| t.shape.get(*axis).copied())
                    .ok_or(format!("no .{suffix} with an extent {axis}"))
            };
            Ok(vec![extent(rows)?, extent(cols)?])
        }
        Logical::Stated(suffix) => {
            let Some((key, t)) = present.get(suffix.as_str()) else {
                return Err(format!("no .{suffix}"));
            };
            let bytes = read(key, t).map_err(|e| e.detail)?;
            let stated: Vec<i64> = match t.dtype {
                Dtype::I64 => (bytes.as_chunks::<8>().0.iter())
                    .map(|b| i64::from_le_bytes(*b))
                    .collect(),
                Dtype::I32 => (bytes.as_chunks::<4>().0.iter())
                    .map(|b| i32::from_le_bytes(*b).into())
                    .collect(),
                _ => return Err(format!(".{suffix} is {}, not I32 or I64", t.dtype)),
            };
            (stated.iter().map(|e| u64::try_from(*e)))
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| format!(".{suffix} states {stated:?}"))
        }
    }
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

/// The format name a producer's own config declares: its carriers' `formats`.
fn format_of(producer: Producer) -> &'static str {
    match producer {
        Producer::Gptq => "gptq",
        Producer::Awq => "awq",
        _ => "",
    }
}

/// A safetensors metadata value as the JSON its producer meant (`"4"`, `"True"`).
fn metadata_value(v: &str) -> Json {
    match v {
        "True" | "true" => Json::Bool(true),
        "False" | "false" => Json::Bool(false),
        _ => v
            .parse::<i64>()
            .map_or_else(|_| Json::String(v.into()), Json::from),
    }
}

/// The one value every config that states a setting gives it (`keys`: the producers' names for
/// it), or why not.
fn stated<'d>(d: &'d Declared, keys: &[&str]) -> std::result::Result<Option<&'d Json>, String> {
    let mut found: Option<(&str, &str, &Json)> = None;
    for (what, doc) in &d.configs {
        for key in keys {
            match (doc.get(*key), found) {
                (Some(v), Some((w, k, f))) if f != v => {
                    return Err(format!("{w} states {k} {f}, {what} states {key} {v}"))
                }
                (Some(v), None) => found = Some((what, key, v)),
                _ => {}
            }
        }
    }
    Ok(found.map(|(_, _, v)| v))
}

/// compressed-tensors' weights args of its `pack-quantized` config groups: one set, or why not
/// (a module's group would need its `targets` matched).
fn ct_weights(d: &Declared) -> std::result::Result<serde_json::Map<String, Json>, String> {
    const READ: [&str; 5] = ["num_bits", "type", "strategy", "group_size", "symmetric"];
    let mut found: Vec<serde_json::Map<String, Json>> = Vec::new();
    for (_, doc) in &d.configs {
        let groups = doc.get("config_groups").and_then(Json::as_object);
        for g in groups.into_iter().flat_map(|g| g.values()) {
            let format = g.get("format").or_else(|| doc.get("format"));
            let Some(w) = g.get("weights").and_then(Json::as_object) else {
                continue;
            };
            if format.and_then(Json::as_str) == Some("pack-quantized") {
                let read: serde_json::Map<String, Json> = READ
                    .iter()
                    .filter_map(|k| Some((k.to_string(), w.get(*k)?.clone())))
                    .collect();
                if !found.contains(&read) {
                    found.push(read);
                }
            }
        }
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err("no pack-quantized config group states its weights".into()),
        _ => Err(format!("pack-quantized groups state different weights ({found:?}); module targets are not read")),
    }
}

/// The code width and params a producer's quantization config states (compressed-tensors'
/// `channel` leaves `group_size` to each module's row length), or why it cannot be read.
pub type Configured = (i64, Vec<(&'static str, Value)>);

fn producer_params(producer: Producer, d: &Declared) -> std::result::Result<Configured, String> {
    let int = |v: Option<&Json>, what: &str| {
        v.and_then(Json::as_i64)
            .ok_or(format!("{what} is not stated as an integer"))
    };
    let flag = |v: Option<&Json>, what: &str| {
        v.and_then(Json::as_bool)
            .ok_or(format!("{what} is not stated as a boolean"))
    };
    Ok(match producer {
        Producer::Gptq => {
            let dynamic = stated(d, &["dynamic"])?;
            if dynamic.is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty()))
            {
                return Err("per-module overrides (dynamic) are not read".into());
            }
            let zero_offset = match stated(d, &["checkpoint_format", "format"])? {
                None => 1u64,
                Some(v) if v == "gptq" => 1,
                Some(v) if v == "gptq_v2" => 0,
                Some(v) => {
                    return Err(format!(
                        "GPTQ format {v} is not read (gptq and gptq_v2 are)"
                    ))
                }
            };
            if let Some(p) = stated(d, &["pack_dtype"])?.filter(|p| *p != "int32") {
                return Err(format!("GPTQ pack_dtype {p} is not read (int32 is)"));
            }
            let params = vec![
                (
                    "group_size",
                    int(stated(d, &["group_size"])?, "group_size")?.into(),
                ),
                ("symmetric", flag(stated(d, &["sym"])?, "sym")?.into()),
                (
                    "act_order",
                    flag(stated(d, &["desc_act"])?, "desc_act")?.into(),
                ),
                ("zero_offset", zero_offset.into()),
            ];
            (int(stated(d, &["bits"])?, "bits")?, params)
        }
        Producer::Awq => {
            let version = stated(d, &["version", "format"])?.and_then(Json::as_str);
            if !version.is_some_and(|v| v.eq_ignore_ascii_case("gemm")) {
                return Err(format!("AWQ version {version:?} is not read (GEMM is)"));
            }
            let zero_point = stated(d, &["zero_point"])?.and_then(Json::as_bool);
            let sym = stated(d, &["sym"])?.and_then(Json::as_bool);
            if zero_point != Some(true) && sym != Some(false) {
                return Err("AWQ without a zero point is not read".into());
            }
            let g = int(stated(d, &["group_size", "q_group_size"])?, "group_size")?;
            (
                int(stated(d, &["bits", "w_bit"])?, "bits")?,
                vec![("group_size", g.into())],
            )
        }
        Producer::CompressedTensors => {
            let w = ct_weights(d)?;
            if w.get("type") != Some(&Json::from("int")) {
                return Err(format!(
                    "pack-quantized weights of type {:?} are not read",
                    w.get("type")
                ));
            }
            let strategy = w.get("strategy").and_then(Json::as_str).unwrap_or_default();
            let mut params = vec![
                ("strategy", strategy.into()),
                ("symmetric", flag(w.get("symmetric"), "symmetric")?.into()),
            ];
            match strategy {
                "group" => {
                    params.push(("group_size", int(w.get("group_size"), "group_size")?.into()))
                }
                "channel" => {}
                other => return Err(format!("pack-quantized strategy {other:?} is not read")),
            }
            (int(w.get("num_bits"), "num_bits")?, params)
        }
        _ => return Err("this producer states no config params".into()),
    })
}

/// A `from_config` carrier's encoding for one module: the config's width and params, the
/// module's row length as a `channel` group, and the packing compressed-tensors' words fit; the
/// scheme checks every part.
fn config_encoding(
    carrier: &Carrier,
    (bits, params): &Configured,
    shape: &[u64],
    parts: &[FoldedPart],
) -> std::result::Result<Encoding, String> {
    let (bits, mut params) = (*bits, params.clone());
    if params
        .iter()
        .any(|(k, v)| *k == "strategy" && *v == Value::from("channel"))
    {
        params.push((
            "group_size",
            (shape.last().copied().unwrap_or(0) as i64).into(),
        ));
    }
    let scheme = carrier.encoding.scheme.replace("{bits}", &bits.to_string());
    let view: BTreeMap<&str, PartBytes> = (parts.iter())
        .map(|p| {
            (
                p.role.as_str(),
                PartBytes {
                    dtype: &p.dtype,
                    shape: &p.shape,
                    bytes: &[],
                },
            )
        })
        .collect();
    let fits = |params: &[(&str, Value)]| {
        let encoding = Encoding {
            scheme: scheme.clone(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        };
        check(&Dtype::F32, shape, &encoding, &view)
            .map(|()| encoding)
            .map_err(|e| e.to_string())
    };
    if carrier.producer != Producer::CompressedTensors {
        return fits(&params);
    }
    // Before compressed-tensors 0.17 the codes filled ⌊32/b⌋ to a word; since, one dense stream.
    // Where `b` divides 32 the bytes are the same, and `dense` names them.
    params.push(("packing", "dense".into()));
    let dense = fits(&params);
    if 32 % bits == 0 {
        return dense;
    }
    params.last_mut().expect("packing").1 = "word".into();
    match (dense, fits(&params)) {
        (Ok(e), Err(_)) | (Err(_), Ok(e)) => Ok(e),
        (Ok(_), Ok(_)) => Err(format!("{bits}-bit words fit both packings")),
        (Err(e), Err(_)) => Err(e),
    }
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
        let folded = match d.producer {
            Producer::Bitsandbytes => {
                bnb::fold_group(set, group, &|c| logical_dtype(group, &d, c))?
            }
            Producer::Nunchaku => nunchaku::fold_group(group, &d, &|c| logical_dtype(group, &d, c))?,
            _ => fold_group(set, group, &d)?,
        };
        let group = &mut set.groups[g];
        for key in &d.markers {
            group.tensors.remove(key);
        }
        // Every consumed key goes before any folded one lands (Nunchaku's bias keeps its key).
        for c in folded.values().flat_map(|(consumed, _)| consumed) {
            group.tensors.remove(c);
        }
        for (key, (_, t)) in folded {
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

fn fold_group(set: &SourceSet, group: &Group, d: &Declared) -> Result<Folds> {
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
    // A producer that states its params in a config states them once for every module.
    let config_rows = rows
        .iter()
        .filter(|c| c.from_config && c.formats.iter().any(|f| d.formats.contains(f)));
    let configured = match config_rows.map(|c| c.producer).next() {
        Some(producer) => match producer_params(producer, d) {
            Ok(c) => Some(c),
            Err(e) => {
                return refuse(
                    Code::UNREGISTERED_ENCODING,
                    format!(
                        "{label}: {} declares a quantization this import does not read: {e}",
                        d.found.join(", ")
                    ),
                )
            }
        },
        None => None,
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
        let stated = conf
            .and_then(|c| conf_get(c, "convrot"))
            .and_then(Json::as_bool);
        let convrot = stated.unwrap_or(false);
        let candidates: Vec<&&Carrier> = rows
            .iter()
            .filter(|c| match c.rotated {
                true => stated != Some(false),
                false => c.convrot == convrot,
            })
            .filter(|c| c.formats.iter().any(|f| formats.contains(f)))
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
                    match stated {
                        Some(true) => " with convrot",
                        Some(false) => " with convrot: false",
                        None => "",
                    },
                    found()
                ),
            );
        }
        let mut fits = Vec::new();
        let mut why = Vec::new();
        let read = |key: &str, t: &Tensor| bytes(set, key, t);
        for c in &candidates {
            match fit(c, &present, conf, &d.config, configured.as_ref(), &read) {
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
        let mut shape = folded.shape.clone();
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
            derive: None,
        };
        out.insert(key, (consumed, tensor));
    }
    Ok(out)
}
