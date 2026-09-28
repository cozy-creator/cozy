//! Exact derived-manifest composition (tfs-019).
//!
//! This is a storage transaction, not a tensor transform. The caller declares a closed
//! component map; this module copies unchanged tensor metadata and ObjectRefs, drops whole
//! logical tensors (and therefore all of their attached roles), and streams only declared
//! additions through the ordinary object writer. There is no filename inference, model
//! vocabulary, Torch, schedule, publication, or serving decision here.

mod adoption;
mod estimate;
pub use estimate::{estimate_write, WriteEstimate};
mod progress;
mod retention;
mod roots;
pub use adoption::adopt_checkpoint;
use progress::read as read_progress;
pub use retention::{release_retention, retain_result, RetainedResult};

use std::collections::{BTreeMap, HashSet};
use std::io::Read;

use crate::canon::{as_arr, as_obj, as_str, Fields, Value};
use crate::checkpoint::{self, build_manifest};
use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Asset, Body, Closure, Header, Part, Tensor};
use crate::ids::{ascii_name, prefixed, ObjectRef};
use crate::limits;
use crate::manifest::{Draft, Entry, Manifest};
use crate::meta::{Hold, Meta};
use crate::read::{self, ReadLease};
use crate::store::Store;
use crate::{registry, spec::EncodingSpec, vectors::EncodingVectors};

fn arithmetic(what: impl Into<String>) -> Refusal {
    Refusal {
        code: Code::ARITH_OVERFLOW,
        detail: what.into(),
    }
}

fn transaction_id(id: &str) -> Result<String> {
    prefixed("ArtifactTransactionId", id)
}

fn name(what: &str, value: &str) -> Result<()> {
    ascii_name(what, value, limits::MAX_NAME_BYTES)
}

fn key(what: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > limits::MAX_KEY_BYTES {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("{what}: key length out of bounds"),
        );
    }
    if !value.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("{what}: key is not printable ASCII"),
        );
    }
    Ok(())
}

fn shape(v: &Value, what: &str) -> Result<Vec<u64>> {
    let values = as_arr(what, "shape", v)?;
    if values.len() > limits::MAX_RANK {
        return refuse(
            Code::RANK_CAP,
            format!("{what}: rank {} over cap", values.len()),
        );
    }
    values
        .iter()
        .map(|v| crate::canon::as_uint(what, "shape", v))
        .collect()
}

fn shape_value(v: &[u64]) -> Value {
    Value::arr(v.iter().map(|n| Value::uint(*n)).collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub alias: String,
    pub manifest: ObjectRef,
}

/// Closed semantic selection from a granted source; never an author-supplied ObjectRef.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartSource {
    pub source: String,
    pub component: String,
    pub tensor: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartDeclaration {
    pub role: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub source: Option<PartSource>,
}

impl PartDeclaration {
    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("dtype", Value::str(self.dtype.name())),
            ("shape", shape_value(&self.shape)),
        ];
        if let Some(selected) = &self.source {
            fields.push((
                "source",
                Value::obj(vec![
                    ("source", Value::str(selected.source.clone())),
                    ("component", Value::str(selected.component.clone())),
                    ("tensor", Value::str(selected.tensor.clone())),
                    ("role", Value::str(selected.role.clone())),
                ]),
            ));
        }
        Value::obj(fields)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDeclaration {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub encoding: String,
    pub parts: Vec<PartDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentDeclaration {
    pub target: String,
    /// Both fields are present or absent. Present means copy/derive from this exact source
    /// component. Absent means create from the complete `add` declaration.
    pub source: Option<String>,
    pub source_component: Option<String>,
    pub drop: Vec<String>,
    pub add: Vec<TensorDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigDeclaration {
    Copy {
        target: String,
        source: String,
        source_config: String,
    },
    Add {
        target: String,
    },
    Derive {
        target: String,
        source: String,
        source_config: String,
    },
}

impl ConfigDeclaration {
    fn target(&self) -> &str {
        match self {
            ConfigDeclaration::Copy { target, .. }
            | ConfigDeclaration::Add { target, .. }
            | ConfigDeclaration::Derive { target, .. } => target,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    /// Required to begin work. Historical declarations remain readable for committed lookup.
    pub work_fingerprint: Option<String>,
    pub sources: Vec<Source>,
    pub components: Vec<ComponentDeclaration>,
    pub configs: Vec<ConfigDeclaration>,
    /// Bounded immutable ordinary companion files, locked into the work intent.
    pub files: Vec<(String, Vec<u8>)>,
    /// Companion files that are already verified Store objects (a pinned tokenizer), named
    /// by reference: no bytes in the intent, none new in the output.
    pub objects: Vec<(String, ObjectRef)>,
    pub order: Vec<(String, String)>,
    pub max_new_bytes: u64,
}

impl Declaration {
    /// One native encoder for the intent, writer and recovery validator.
    pub fn canonical_work_bytes(&mut self) -> Result<Vec<u8>> {
        self.normalize_and_validate()?;
        if self.work_fingerprint.is_none() {
            return refuse(
                Code::MISSING_FIELD,
                "derived work requires an explicit producer fingerprint",
            );
        }
        self.require_order()?;
        let bytes = crate::canon::write(&self.to_value());
        if bytes.len() > limits::DOC_MAX_BYTES / 2 {
            return refuse(
                Code::SIZE_CAP,
                "derived declaration is over its metadata budget",
            );
        }
        Ok(bytes)
    }

    fn require_order(&self) -> Result<()> {
        if self.order.is_empty() {
            return refuse(
                Code::CONSTRUCTION_ORDER_REQUIRED,
                "derived artifact declaration has no construction traversal",
            );
        }
        let mut seen = HashSet::new();
        for (component, tensor_key) in &self.order {
            name("construction-order component", component)?;
            key("construction-order tensor", tensor_key)?;
            if !seen.insert((component.as_str(), tensor_key.as_str())) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("construction order repeats {component}/{tensor_key}"),
                );
            }
        }
        Ok(())
    }

    pub fn normalize_and_validate(&mut self) -> Result<()> {
        if let Some(fingerprint) = &self.work_fingerprint {
            prefixed("producer work fingerprint", fingerprint)?;
        }
        if self.components.is_empty() || self.components.len() > limits::MAX_COMPONENTS {
            return refuse(
                Code::EMPTY_COMPONENTS,
                format!(
                    "target components count {} out of bounds",
                    self.components.len()
                ),
            );
        }
        if self.max_new_bytes > limits::INT_MAX as u64 {
            return refuse(
                Code::NUMBER_RANGE,
                "max_new_bytes is outside the canonical integer range",
            );
        }

        validate_files(&mut self.files)?;
        self.objects.sort_by(|a, b| a.0.cmp(&b.0));
        let mut paths: Vec<String> = self
            .files
            .iter()
            .map(|(path, _)| path.clone())
            .chain(std::iter::once("model.cozytensors".to_string()))
            .collect();
        for (path, object) in &self.objects {
            crate::manifest::check_path(path)?;
            crate::ids::hex64("companion object", &object.sha256)?;
            paths.push(path.clone());
        }
        validate_companion_paths(&mut paths)?;

        for source in &self.sources {
            name("source alias", &source.alias)?;
            if source.manifest.length == 0 {
                return refuse(Code::ZERO_ELEMENT, "source manifest length is zero");
            }
            crate::ids::hex64("source manifest", &source.manifest.sha256)?;
        }
        self.sources.sort_by(|a, b| a.alias.cmp(&b.alias));
        if let Some(pair) = self.sources.windows(2).find(|w| w[0].alias == w[1].alias) {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("source alias {:?} appears twice", pair[0].alias),
            );
        }
        let source_names: HashSet<&str> = self.sources.iter().map(|s| s.alias.as_str()).collect();

        for component in &mut self.components {
            name("target component", &component.target)?;
            match (&component.source, &component.source_component) {
                (Some(source), Some(source_component)) => {
                    if !source_names.contains(source.as_str()) {
                        return refuse(
                            Code::ROOT_ABSENT,
                            format!(
                                "target {:?} selects undeclared source alias {source:?}",
                                component.target
                            ),
                        );
                    }
                    name("source component", source_component)?;
                }
                (None, None) => {
                    if !component.drop.is_empty() {
                        return refuse(
                            Code::MISSING_COMPONENT,
                            format!(
                                "created target {:?} has drop keys but no source component",
                                component.target
                            ),
                        );
                    }
                    if component.add.is_empty() {
                        return refuse(
                            Code::EMPTY_COMPONENTS,
                            format!("created target {:?} declares no tensors", component.target),
                        );
                    }
                }
                _ => {
                    return refuse(
                        Code::MISSING_FIELD,
                        format!(
                            "target {:?} must declare source and source_component together",
                            component.target
                        ),
                    )
                }
            }

            for drop in &component.drop {
                key("drop", drop)?;
            }
            component.drop.sort();
            if let Some(pair) = component.drop.windows(2).find(|w| w[0] == w[1]) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("target {:?} drops {:?} twice", component.target, pair[0]),
                );
            }

            for tensor in &mut component.add {
                key("addition", &tensor.key)?;
                prefixed("addition encoding", &tensor.encoding)?;
                checked_bytes(
                    &format!("{}/{}", component.target, tensor.key),
                    &tensor.shape,
                    tensor.dtype,
                )?;
                if tensor.parts.is_empty() || tensor.parts.len() > limits::MAX_PARTS {
                    return refuse(
                        Code::COUNT_CAP,
                        format!(
                            "{}/{} declares {} parts out of bounds",
                            component.target,
                            tensor.key,
                            tensor.parts.len()
                        ),
                    );
                }
                for part in &tensor.parts {
                    name("addition role", &part.role)?;
                    if let Some(selected) = &part.source {
                        name("part source alias", &selected.source)?;
                        name("part source component", &selected.component)?;
                        key("part source tensor", &selected.tensor)?;
                        name("part source role", &selected.role)?;
                        if !source_names.contains(selected.source.as_str()) {
                            return refuse(
                                Code::ROOT_ABSENT,
                                "part selects undeclared source alias",
                            );
                        }
                        if selected.role != part.role {
                            return refuse(
                                Code::TRANSACTION_CONFLICT,
                                "grafted part must preserve its encoding role",
                            );
                        }
                    }
                    checked_bytes(
                        &format!("{}/{}#{}", component.target, tensor.key, part.role),
                        &part.shape,
                        part.dtype,
                    )?;
                }
                tensor.parts.sort_by(|a, b| a.role.cmp(&b.role));
                if let Some(pair) = tensor.parts.windows(2).find(|w| w[0].role == w[1].role) {
                    return refuse(
                        Code::DUPLICATE_KEY,
                        format!(
                            "{}/{} declares role {:?} twice",
                            component.target, tensor.key, pair[0].role
                        ),
                    );
                }
            }
            let mut add_keys: Vec<&str> = component
                .add
                .iter()
                .map(|tensor| tensor.key.as_str())
                .collect();
            add_keys.sort();
            if let Some(pair) = add_keys.windows(2).find(|w| w[0] == w[1]) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("target {:?} adds key {:?} twice", component.target, pair[0]),
                );
            }
        }
        let mut component_names: Vec<&str> = self
            .components
            .iter()
            .map(|component| component.target.as_str())
            .collect();
        component_names.sort();
        if let Some(pair) = component_names.windows(2).find(|w| w[0] == w[1]) {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("target component {:?} appears twice", pair[0]),
            );
        }

        for config in &self.configs {
            name("target config", config.target())?;
            match config {
                ConfigDeclaration::Copy {
                    source,
                    source_config,
                    ..
                } => {
                    if !source_names.contains(source.as_str()) {
                        return refuse(
                            Code::ROOT_ABSENT,
                            format!("config selects undeclared source alias {source:?}"),
                        );
                    }
                    name("source config", source_config)?;
                }
                ConfigDeclaration::Add { .. } => {}
                ConfigDeclaration::Derive {
                    source,
                    source_config,
                    ..
                } => {
                    if !source_names.contains(source.as_str()) {
                        return refuse(
                            Code::ROOT_ABSENT,
                            format!("config selects undeclared source alias {source:?}"),
                        );
                    }
                    name("source config", source_config)?;
                }
            }
        }
        self.configs.sort_by(|a, b| a.target().cmp(b.target()));
        if let Some(pair) = self
            .configs
            .windows(2)
            .find(|w| w[0].target() == w[1].target())
        {
            return refuse(
                Code::DUPLICATE_KEY,
                format!("target config {:?} appears twice", pair[0].target()),
            );
        }

        let used_sources: HashSet<&str> = self
            .components
            .iter()
            .filter_map(|component| component.source.as_deref())
            .chain(
                self.components
                    .iter()
                    .flat_map(|component| &component.add)
                    .flat_map(|tensor| &tensor.parts)
                    .filter_map(|part| {
                        part.source
                            .as_ref()
                            .map(|selected| selected.source.as_str())
                    }),
            )
            .chain(self.configs.iter().filter_map(|config| match config {
                ConfigDeclaration::Copy { source, .. }
                | ConfigDeclaration::Derive { source, .. } => Some(source.as_str()),
                ConfigDeclaration::Add { .. } => None,
            }))
            .collect();
        if let Some(unused) = self
            .sources
            .iter()
            .find(|source| !used_sources.contains(source.alias.as_str()))
        {
            return refuse(
                Code::UNKNOWN_FIELD,
                format!(
                    "source alias {:?} is unused; the exact source set is closed and minimal",
                    unused.alias
                ),
            );
        }

        let (bytes, objects) = self.planned_additions()?;
        if bytes > self.max_new_bytes {
            return refuse(
                Code::QUOTA_EXHAUSTED,
                format!(
                    "closed additions require {bytes} bytes, output grant admits {}",
                    self.max_new_bytes
                ),
            );
        }
        if objects > limits::MAX_TOTAL_REFS {
            return refuse(
                Code::TOTAL_REFS_CAP,
                format!(
                    "closed additions require {objects} objects, TensorFS admits at most {}",
                    limits::MAX_TOTAL_REFS
                ),
            );
        }
        Ok(())
    }

    fn planned_additions(&self) -> Result<(u64, usize)> {
        let mut bytes: u64 = self.files.iter().map(|(_, data)| data.len() as u64).sum();
        let mut objects = self.files.len();
        for component in &self.components {
            for tensor in &component.add {
                for part in &tensor.parts {
                    if part.source.is_some() {
                        continue;
                    }
                    let n = checked_bytes(
                        &format!("{}/{}#{}", component.target, tensor.key, part.role),
                        &part.shape,
                        part.dtype,
                    )?;
                    bytes = bytes
                        .checked_add(n)
                        .ok_or_else(|| arithmetic("addition byte total overflows"))?;
                    if !Part::is_inline(n) {
                        let segments = n
                            .checked_add(limits::GRID_BYTES - 1)
                            .ok_or_else(|| arithmetic("addition segment count overflows"))?
                            / limits::GRID_BYTES;
                        objects = objects
                            .checked_add(segments as usize)
                            .ok_or_else(|| arithmetic("addition object count overflows"))?;
                    }
                }
            }
        }
        Ok((bytes, objects))
    }

    fn canonical_text(&self) -> String {
        String::from_utf8(crate::canon::write(&self.to_value())).expect("canonical JSON is UTF-8")
    }

    fn parse_text(text: &str) -> Result<Declaration> {
        let v = crate::canon::parse_canonical(text.as_bytes(), limits::DOC_MAX_BYTES)?;
        let mut d = Declaration::from_value(&v)?;
        d.normalize_and_validate()?;
        if d.canonical_text() != text {
            return refuse(
                Code::NONCANONICAL_ENCODING,
                "derived declaration is not in normalized canonical order",
            );
        }
        Ok(d)
    }

    fn from_value(v: &Value) -> Result<Declaration> {
        let mut f = Fields::new("DerivedDeclaration", v)?;
        let max_new_bytes = f.req_uint("max_new_bytes")?;
        let mut files = Vec::new();
        if let Some(value) = f.opt("files") {
            for (path, value) in as_obj("DerivedDeclaration", "files", value)? {
                files.push((
                    path.clone(),
                    decode_file(as_str("companion", "data", value)?)?,
                ));
            }
        }
        let mut objects = Vec::new();
        if let Some(value) = f.opt("objects") {
            for (path, value) in as_obj("DerivedDeclaration", "objects", value)? {
                objects.push((
                    path.clone(),
                    ObjectRef::from_value("companion object", value)?,
                ));
            }
        }
        let work_fingerprint = f
            .opt("work_fingerprint")
            .map(|value| {
                as_str("DerivedDeclaration", "work_fingerprint", value).map(str::to_string)
            })
            .transpose()?;

        let mut sources = Vec::new();
        for (alias, source_value) in as_obj("DerivedDeclaration", "sources", f.req("sources")?)? {
            let mut sf = Fields::new("DerivedDeclaration.source", source_value)?;
            let manifest =
                ObjectRef::from_value("DerivedDeclaration.source.manifest", sf.req("manifest")?)?;
            sf.done()?;
            sources.push(Source {
                alias: alias.clone(),
                manifest,
            });
        }

        let mut components = Vec::new();
        for (target, component_value) in
            as_obj("DerivedDeclaration", "components", f.req("components")?)?
        {
            let mut cf = Fields::new("DerivedDeclaration.component", component_value)?;
            let source = cf
                .opt("source")
                .map(|v| as_str("component", "source", v).map(str::to_string))
                .transpose()?;
            let source_component = cf
                .opt("source_component")
                .map(|v| as_str("component", "source_component", v).map(str::to_string))
                .transpose()?;
            let mut drop = Vec::new();
            for v in as_arr("component", "drop", cf.req("drop")?)? {
                drop.push(as_str("component", "drop", v)?.to_string());
            }
            let mut add = Vec::new();
            for (tensor_key, tensor_value) in as_obj("component", "add", cf.req("add")?)? {
                let mut tf = Fields::new("DerivedDeclaration.tensor", tensor_value)?;
                let encoding = prefixed("tensor encoding", tf.req_str("encoding")?)?;
                let mut lf = Fields::new("DerivedDeclaration.logical", tf.req("logical")?)?;
                let dtype = Dtype::parse(lf.req_str("dtype")?)?;
                let logical_shape = shape(lf.req("shape")?, "DerivedDeclaration.logical")?;
                lf.done()?;
                let mut parts = Vec::new();
                for (role, part_value) in as_obj("tensor", "parts", tf.req("parts")?)? {
                    let mut pf = Fields::new("DerivedDeclaration.part", part_value)?;
                    let dtype = Dtype::parse(pf.req_str("dtype")?)?;
                    let part_shape = shape(pf.req("shape")?, "DerivedDeclaration.part")?;
                    let source = pf
                        .opt("source")
                        .map(|value| {
                            let mut sf = Fields::new("DerivedDeclaration.part.source", value)?;
                            let selected = PartSource {
                                source: sf.req_str("source")?.to_string(),
                                component: sf.req_str("component")?.to_string(),
                                tensor: sf.req_str("tensor")?.to_string(),
                                role: sf.req_str("role")?.to_string(),
                            };
                            sf.done()?;
                            Ok(selected)
                        })
                        .transpose()?;
                    pf.done()?;
                    parts.push(PartDeclaration {
                        role: role.clone(),
                        dtype,
                        shape: part_shape,
                        source,
                    });
                }
                tf.done()?;
                add.push(TensorDeclaration {
                    key: tensor_key.clone(),
                    dtype,
                    shape: logical_shape,
                    encoding,
                    parts,
                });
            }
            cf.done()?;
            components.push(ComponentDeclaration {
                target: target.clone(),
                source,
                source_component,
                drop,
                add,
            });
        }

        let mut configs = Vec::new();
        for (target, config_value) in as_obj("DerivedDeclaration", "configs", f.req("configs")?)? {
            let mut cf = Fields::new("DerivedDeclaration.config", config_value)?;
            let kind = cf.req_str("kind")?;
            let source = cf.opt("source");
            let source_config = cf.opt("source_config");
            let config = match (kind, source, source_config) {
                ("copy", Some(source), Some(source_config)) => ConfigDeclaration::Copy {
                    target: target.clone(),
                    source: as_str("config", "source", source)?.to_string(),
                    source_config: as_str("config", "source_config", source_config)?.to_string(),
                },
                ("add", None, None) => ConfigDeclaration::Add {
                    target: target.clone(),
                },
                ("derive", Some(source), Some(source_config)) => ConfigDeclaration::Derive {
                    target: target.clone(),
                    source: as_str("config", "source", source)?.to_string(),
                    source_config: as_str("config", "source_config", source_config)?.to_string(),
                },
                _ => {
                    return refuse(
                        Code::MISSING_FIELD,
                        format!("config {target:?} has an invalid kind/source combination"),
                    )
                }
            };
            cf.done()?;
            configs.push(config);
        }
        let mut order = Vec::new();
        for (index, value) in as_arr("DerivedDeclaration", "order", f.req("order")?)?
            .iter()
            .enumerate()
        {
            let row = as_arr("DerivedDeclaration.order", "row", value)?;
            if row.len() != 2 {
                return refuse(
                    Code::WRONG_TYPE,
                    format!(
                        "DerivedDeclaration.order[{index}] arity {}, expected 2",
                        row.len()
                    ),
                );
            }
            order.push((
                as_str("DerivedDeclaration.order", "component", &row[0])?.to_string(),
                as_str("DerivedDeclaration.order", "key", &row[1])?.to_string(),
            ));
        }
        f.done()?;
        Ok(Declaration {
            work_fingerprint,
            sources,
            components,
            configs,
            files,
            objects,
            order,
            max_new_bytes,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "components",
                Value::map(
                    self.components
                        .iter()
                        .map(|component| {
                            let mut fields = vec![
                                (
                                    "add".to_string(),
                                    Value::map(
                                        component
                                            .add
                                            .iter()
                                            .map(|tensor| {
                                                (
                                                    tensor.key.clone(),
                                                    Value::obj(vec![
                                                        (
                                                            "encoding",
                                                            Value::str(tensor.encoding.clone()),
                                                        ),
                                                        (
                                                            "logical",
                                                            Value::obj(vec![
                                                                (
                                                                    "dtype",
                                                                    Value::str(tensor.dtype.name()),
                                                                ),
                                                                (
                                                                    "shape",
                                                                    shape_value(&tensor.shape),
                                                                ),
                                                            ]),
                                                        ),
                                                        (
                                                            "parts",
                                                            Value::map(
                                                                tensor
                                                                    .parts
                                                                    .iter()
                                                                    .map(|part| {
                                                                        (
                                                                            part.role.clone(),
                                                                            part.to_value(),
                                                                        )
                                                                    })
                                                                    .collect(),
                                                            ),
                                                        ),
                                                    ]),
                                                )
                                            })
                                            .collect(),
                                    ),
                                ),
                                (
                                    "drop".to_string(),
                                    Value::arr(
                                        component
                                            .drop
                                            .iter()
                                            .map(|key| Value::str(key.clone()))
                                            .collect(),
                                    ),
                                ),
                            ];
                            if let Some(source) = &component.source {
                                fields.push(("source".to_string(), Value::str(source.clone())));
                            }
                            if let Some(source_component) = &component.source_component {
                                fields.push((
                                    "source_component".to_string(),
                                    Value::str(source_component.clone()),
                                ));
                            }
                            (component.target.clone(), Value::map(fields))
                        })
                        .collect(),
                ),
            ),
            (
                "configs",
                Value::map(
                    self.configs
                        .iter()
                        .map(|config| match config {
                            ConfigDeclaration::Copy {
                                target,
                                source,
                                source_config,
                            } => (
                                target.clone(),
                                Value::obj(vec![
                                    ("kind", Value::str("copy")),
                                    ("source", Value::str(source.clone())),
                                    ("source_config", Value::str(source_config.clone())),
                                ]),
                            ),
                            ConfigDeclaration::Add { target } => (
                                target.clone(),
                                Value::obj(vec![("kind", Value::str("add"))]),
                            ),
                            ConfigDeclaration::Derive {
                                target,
                                source,
                                source_config,
                            } => (
                                target.clone(),
                                Value::obj(vec![
                                    ("kind", Value::str("derive")),
                                    ("source", Value::str(source.clone())),
                                    ("source_config", Value::str(source_config.clone())),
                                ]),
                            ),
                        })
                        .collect(),
                ),
            ),
            ("max_new_bytes", Value::uint(self.max_new_bytes)),
            (
                "order",
                Value::arr(
                    self.order
                        .iter()
                        .map(|(component, key)| {
                            Value::arr(vec![Value::str(component.clone()), Value::str(key.clone())])
                        })
                        .collect(),
                ),
            ),
            (
                "sources",
                Value::map(
                    self.sources
                        .iter()
                        .map(|source| {
                            (
                                source.alias.clone(),
                                Value::obj(vec![("manifest", source.manifest.to_value())]),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if !self.files.is_empty() {
            fields.push((
                "files",
                Value::map(
                    self.files
                        .iter()
                        .map(|(path, data)| (path.clone(), Value::str(crate::b64::encode(data))))
                        .collect(),
                ),
            ));
        }
        if !self.objects.is_empty() {
            fields.push((
                "objects",
                Value::map(
                    self.objects
                        .iter()
                        .map(|(path, object)| (path.clone(), object.to_value()))
                        .collect(),
                ),
            ));
        }
        if let Some(fingerprint) = &self.work_fingerprint {
            fields.push(("work_fingerprint", Value::str(fingerprint.clone())));
        }
        Value::obj(fields)
    }
}

/// Companion documents are intentionally small; their bytes live in the immutable intent.
pub const MAX_COMPANION_FILES: usize = 16;
pub const MAX_COMPANION_BYTES: usize = 64 << 10;
/// Leaves declaration space below Runtime's one-MiB transport ceiling after base64.
pub const MAX_COMPANION_TOTAL_BYTES: usize = 512 << 10;

pub fn decode_file(encoded: &str) -> Result<Vec<u8>> {
    if encoded.len() > MAX_COMPANION_BYTES.div_ceil(3) * 4 {
        return refuse(Code::SIZE_CAP, "companion file exceeds 64 KiB");
    }
    let data = crate::b64::decode(encoded)?;
    if data.len() > MAX_COMPANION_BYTES {
        return refuse(Code::SIZE_CAP, "companion file exceeds 64 KiB");
    }
    Ok(data)
}

fn validate_files(files: &mut [(String, Vec<u8>)]) -> Result<()> {
    if files.len() > MAX_COMPANION_FILES {
        return refuse(
            Code::COUNT_CAP,
            "at most 16 companion files may be declared",
        );
    }
    if files.iter().map(|(_, data)| data.len()).sum::<usize>() > MAX_COMPANION_TOTAL_BYTES {
        return refuse(Code::SIZE_CAP, "companion files exceed 512 KiB in total");
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut paths = vec!["model.cozytensors".to_string()];
    for (path, data) in files.iter() {
        crate::manifest::check_path(path)?;
        if data.len() > MAX_COMPANION_BYTES {
            return refuse(Code::SIZE_CAP, "companion file exceeds 64 KiB");
        }
        paths.push(path.clone());
    }
    validate_companion_paths(&mut paths)
}

fn validate_companion_paths(paths: &mut [String]) -> Result<()> {
    paths.sort();
    if paths.windows(2).any(|pair| pair[0] == pair[1]) {
        return refuse(
            Code::PATH_ILLEGAL,
            "companion paths collide with checkpoint or other files",
        );
    }
    let names: HashSet<_> = paths.iter().map(String::as_str).collect();
    for path in paths.iter() {
        for (at, _) in path.match_indices('/') {
            if names.contains(&path[..at]) {
                return refuse(Code::PATH_ILLEGAL, "companion path traverses another file");
            }
        }
    }
    Ok(())
}

fn companion_files(
    declaration: &Declaration,
    sources: &[SourceData],
) -> Result<BTreeMap<String, Entry>> {
    let mut files = BTreeMap::new();
    let explicit: HashSet<_> = declaration
        .files
        .iter()
        .map(|(path, _)| path.as_str())
        .chain(declaration.objects.iter().map(|(path, _)| path.as_str()))
        .collect();
    for source in sources {
        let mut backing: HashSet<_> = source
            .header
            .tensors()
            .flat_map(|(_, _, tensor)| tensor.parts.iter())
            .flat_map(|(_, part)| part.segments())
            .map(|o| (o.sha256.clone(), o.length))
            .collect();
        backing.insert((source.fact.header.sha256.clone(), source.fact.header.length));
        for (path, entry) in source.manifest.entries() {
            // Entries a newer TensorFS wrote travel with their kind unchanged.
            let Some(object) = entry.content() else {
                continue;
            };
            // A named asset is an explicit semantic role, even when CAS deduplication
            // makes its bytes identical to a tensor part. Only unindexed backing aliases
            // disappear from the derived manifest.
            let named_asset = source.header.assets.iter().any(|(name, asset)| {
                name == path
                    && asset.logical_sha256 == object.sha256
                    && asset.logical_length == object.length
            });
            if (backing.contains(&(object.sha256.clone(), object.length)) && !named_asset)
                || explicit.contains(path.as_str())
            {
                continue;
            }
            if let Some(previous) = files.insert(path.clone(), entry.clone()) {
                if previous != *entry {
                    return refuse(Code::TRANSACTION_CONFLICT,
                        format!("sources disagree on companion {path:?}; explicitly declare replacement bytes"));
                }
            }
        }
    }
    files.extend(
        declaration
            .files
            .iter()
            .map(|(path, data)| (path.clone(), Entry::File(ObjectRef::of(data)))),
    );
    files.extend(
        declaration
            .objects
            .iter()
            .map(|(path, object)| (path.clone(), Entry::File(object.clone()))),
    );
    let mut paths: Vec<_> = files
        .keys()
        .cloned()
        .chain(std::iter::once("model.cozytensors".into()))
        .collect();
    validate_companion_paths(&mut paths)?;
    Ok(files)
}

fn selected_components(declaration: &Declaration, source_alias: &str) -> Vec<String> {
    let mut components: Vec<String> = declaration
        .components
        .iter()
        .filter_map(
            |component| match (&component.source, &component.source_component) {
                (Some(alias), Some(source_component)) if alias == source_alias => {
                    Some(source_component.clone())
                }
                _ => None,
            },
        )
        .collect();
    components.extend(
        declaration
            .components
            .iter()
            .flat_map(|component| &component.add)
            .flat_map(|tensor| &tensor.parts)
            .filter_map(|part| part.source.as_ref())
            .filter(|selected| selected.source == source_alias)
            .map(|selected| selected.component.clone()),
    );
    components.sort();
    components.dedup();
    components
}

fn selected_configs(declaration: &Declaration, source_alias: &str) -> Vec<String> {
    let mut configs: Vec<String> = declaration
        .configs
        .iter()
        .filter_map(|config| match config {
            ConfigDeclaration::Copy {
                source,
                source_config,
                ..
            }
            | ConfigDeclaration::Derive {
                source,
                source_config,
                ..
            } if source == source_alias => Some(source_config.clone()),
            _ => None,
        })
        .collect();
    configs.sort();
    configs.dedup();
    configs
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AddedPart {
    pub(crate) component: String,
    pub(crate) key: String,
    pub(crate) role: String,
    pub(crate) part: Part,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AddedConfig {
    pub(crate) name: String,
    pub(crate) config: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransactionState {
    Open,
    Committed(Disposition),
    Abandoned,
}

impl TransactionState {
    fn name(&self) -> &'static str {
        match self {
            TransactionState::Open => "open",
            TransactionState::Committed(_) => "committed",
            TransactionState::Abandoned => "abandoned",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Disposition {
    Pending,
    Adopted(String),
    Released,
}

impl Disposition {
    fn to_value(&self) -> Value {
        match self {
            Self::Pending => Value::obj(vec![("kind", Value::str("pending"))]),
            Self::Released => Value::obj(vec![("kind", Value::str("released"))]),
            Self::Adopted(root) => Value::obj(vec![
                ("kind", Value::str("adopted")),
                ("scratch_root_id", Value::str(root.clone())),
            ]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransactionRow {
    pub(crate) id: String,
    pub(crate) declaration: Option<String>,
    pub(crate) state: TransactionState,
    pub(crate) highest_session: u64,
    pub(crate) current_session: Option<u64>,
    pub(crate) added_parts: Vec<AddedPart>,
    pub(crate) added_configs: Vec<AddedConfig>,
    pub(crate) source_facts: Vec<SourceFact>,
    pub(crate) inherited_payload_objects: u64,
    pub(crate) inherited_payload_bytes: u64,
    pub(crate) header: Option<ObjectRef>,
    pub(crate) manifest: Option<ObjectRef>,
}

fn part_value(part: &Part) -> Value {
    let mut fields = vec![
        ("dtype".to_string(), Value::str(part.dtype.name())),
        ("shape".to_string(), shape_value(&part.shape)),
    ];
    match &part.body {
        Body::Inline(value) => {
            fields.push(("inline".to_string(), Value::str(crate::b64::encode(value))))
        }
        Body::Segments(segments) => fields.push((
            "segments".to_string(),
            Value::arr(segments.iter().map(ObjectRef::to_value).collect()),
        )),
    }
    Value::map(fields)
}

fn part_from_value(value: &Value) -> Result<Part> {
    let mut f = Fields::new("DerivedPart", value)?;
    let dtype = Dtype::parse(f.req_str("dtype")?)?;
    let shape = shape(f.req("shape")?, "DerivedPart")?;
    let inline = f.opt("inline");
    let segments = f.opt("segments");
    f.done()?;
    let body = match (inline, segments) {
        (Some(value), None) => {
            Body::Inline(crate::b64::decode(as_str("DerivedPart", "inline", value)?)?)
        }
        (None, Some(values)) => Body::Segments(
            as_arr("DerivedPart", "segments", values)?
                .iter()
                .map(|value| ObjectRef::from_value("DerivedPart.segment", value))
                .collect::<Result<Vec<_>>>()?,
        ),
        _ => {
            return refuse(
                Code::INLINE_EXCLUSIVE,
                "DerivedPart carries exactly one of inline or segments",
            )
        }
    };
    let part = Part { dtype, shape, body };
    part.check_bytes("DerivedPart")?;
    Ok(part)
}

/// The stored identity of one accepted part: the canonical bytes of its value.
pub(crate) fn part_bytes(part: &Part) -> Vec<u8> {
    crate::canon::write(&part_value(part))
}

impl AddedPart {
    pub(crate) fn stored(
        component: String,
        key: String,
        role: String,
        bytes: &[u8],
    ) -> Result<Self> {
        name("added part component", &component)?;
        self::key("added part tensor", &key)?;
        name("added part role", &role)?;
        let part = part_from_value(&crate::canon::parse_canonical(
            bytes,
            limits::DOC_MAX_BYTES,
        )?)?;
        Ok(Self {
            component,
            key,
            role,
            part,
        })
    }

    pub(crate) fn same_value(&self, other: &AddedPart) -> bool {
        (&self.component, &self.key, &self.role) == (&other.component, &other.key, &other.role)
    }

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("DerivedTransaction.added_part", value)?;
        let component = fields.req_str("component")?.to_string();
        let tensor_key = fields.req_str("key")?.to_string();
        let part_value = crate::canon::parse_canonical(
            fields.req_str("part")?.as_bytes(),
            limits::DOC_MAX_BYTES,
        )?;
        let part = part_from_value(&part_value)?;
        let role = fields.req_str("role")?.to_string();
        fields.done()?;
        name("added part component", &component)?;
        key("added part tensor", &tensor_key)?;
        name("added part role", &role)?;
        Ok(Self {
            component,
            key: tensor_key,
            role,
            part,
        })
    }
    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("component", Value::str(self.component.clone())),
            ("key", Value::str(self.key.clone())),
            (
                "part",
                Value::str(
                    String::from_utf8(crate::canon::write(&part_value(&self.part)))
                        .expect("canonical JSON is UTF-8"),
                ),
            ),
            ("role", Value::str(self.role.clone())),
        ])
    }
}

impl AddedConfig {
    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("DerivedTransaction.added_config", value)?;
        let config = fields.req_str("canonical_json")?.as_bytes().to_vec();
        if crate::header::canonical_config("added config", &config)? != config {
            return refuse(
                Code::NONCANONICAL_ENCODING,
                "added config is not canonical RFC 8785 JSON",
            );
        }
        let config_name = fields.req_str("name")?.to_string();
        fields.done()?;
        name("added config", &config_name)?;
        Ok(Self {
            name: config_name,
            config,
        })
    }
    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "canonical_json",
                Value::str(std::str::from_utf8(&self.config).expect("validated config is UTF-8")),
            ),
            ("name", Value::str(self.name.clone())),
        ])
    }
}

impl TransactionRow {
    /// The row's SQL form: accepted parts are rows of their own, and `open_session`
    /// names the writer that may add to it.
    pub(crate) fn stored(&self) -> (Vec<u8>, Option<u64>) {
        let mut row = self.clone();
        row.added_parts.clear();
        let open = match self.state {
            TransactionState::Open => self.current_session,
            _ => None,
        };
        (crate::canon::write(&row.to_value()), open)
    }

    /// Join stored part rows onto a row read back; a row written by an older build may
    /// still carry its parts inline, and the two must agree.
    pub(crate) fn accept_stored(&mut self, parts: Vec<AddedPart>) -> Result<()> {
        for added in parts {
            match self.added_parts.iter().find(|old| old.same_value(&added)) {
                Some(old) if *old != added => {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "stored part conflicts with the transaction row",
                    )
                }
                Some(_) => {}
                None => self.added_parts.push(added),
            }
        }
        self.normalize();
        Ok(())
    }

    pub(crate) fn normalize(&mut self) {
        self.added_parts
            .sort_by(|a, b| (&a.component, &a.key, &a.role).cmp(&(&b.component, &b.key, &b.role)));
        self.added_configs.sort_by(|a, b| a.name.cmp(&b.name));
        self.source_facts.sort_by(|a, b| a.alias.cmp(&b.alias));
    }

    pub(crate) fn from_value(value: &Value) -> Result<TransactionRow> {
        let mut f = Fields::new("DerivedTransaction", value)?;
        let id = transaction_id(f.req_str("id")?)?;
        let state_name = f.req_str("state")?.to_string();
        let highest_session = f.req_uint("highest_session")?;
        let current_session = f
            .opt("current_session")
            .map(|v| crate::canon::as_uint("DerivedTransaction", "current_session", v))
            .transpose()?;
        let declaration = f
            .opt("declaration")
            .map(|v| as_str("DerivedTransaction", "declaration", v).map(str::to_string))
            .transpose()?;

        let added_parts = as_arr("DerivedTransaction", "added_parts", f.req("added_parts")?)?
            .iter()
            .map(AddedPart::from_value)
            .collect::<Result<Vec<_>>>()?;
        let added_configs = as_arr(
            "DerivedTransaction",
            "added_configs",
            f.req("added_configs")?,
        )?
        .iter()
        .map(AddedConfig::from_value)
        .collect::<Result<Vec<_>>>()?;
        let inherited_payload_bytes = f.req_uint("inherited_payload_bytes")?;
        let inherited_payload_objects = f.req_uint("inherited_payload_objects")?;
        let mut source_facts = Vec::new();
        for value in as_arr("DerivedTransaction", "source_facts", f.req("source_facts")?)? {
            let mut sf = Fields::new("DerivedTransaction.source_fact", value)?;
            let alias = sf.req_str("alias")?.to_string();
            let mut components = Vec::new();
            for component in as_arr("source_fact", "components", sf.req("components")?)? {
                components.push(as_str("source_fact", "component", component)?.to_string());
            }
            let header = ObjectRef::from_value("source_fact.header", sf.req("header")?)?;
            let manifest = ObjectRef::from_value("source_fact.manifest", sf.req("manifest")?)?;
            sf.done()?;
            name("source fact alias", &alias)?;
            for component in &components {
                name("source fact component", component)?;
            }
            if components.windows(2).any(|w| w[0] >= w[1]) {
                return refuse(
                    Code::SORT_ORDER,
                    format!("source fact {alias:?} components are not sorted unique"),
                );
            }
            source_facts.push(SourceFact {
                alias,
                manifest,
                header,
                components,
            });
        }
        let header = f
            .opt("header")
            .map(|v| ObjectRef::from_value("DerivedTransaction.header", v))
            .transpose()?;
        let manifest = f
            .opt("manifest")
            .map(|v| ObjectRef::from_value("DerivedTransaction.manifest", v))
            .transpose()?;
        let disposition = match f.opt("disposition") {
            None => None,
            Some(value) => {
                let mut df = Fields::new("DerivedTransaction.disposition", value)?;
                let disposition = match df.req_str("kind")? {
                    "pending" => Disposition::Pending,
                    "released" => Disposition::Released,
                    "adopted" => {
                        let root = df.req_str("scratch_root_id")?;
                        name("private scratch root", root)?;
                        Disposition::Adopted(root.into())
                    }
                    _ => return refuse(Code::UNKNOWN_FIELD, "invalid derived root disposition"),
                };
                df.done()?;
                Some(disposition)
            }
        };
        f.done()?;
        let state = match (state_name.as_str(), disposition) {
            ("open", None) => TransactionState::Open,
            ("committed", Some(disposition)) => TransactionState::Committed(disposition),
            ("abandoned", None) => TransactionState::Abandoned,
            (known @ ("open" | "committed" | "abandoned"), _) => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("derived transaction state {known:?} has an invalid disposition"),
                )
            }
            (other, _) => {
                return refuse(
                    Code::UNKNOWN_FIELD,
                    format!("unknown derived transaction state {other:?}"),
                )
            }
        };

        if highest_session > limits::INT_MAX as u64
            || current_session.is_some_and(|id| id == 0 || id > limits::INT_MAX as u64)
            || current_session.is_some_and(|id| id > highest_session)
        {
            return refuse(
                Code::NUMBER_RANGE,
                "WriterSessionId is outside the canonical integer range",
            );
        }
        if let Some(text) = &declaration {
            let declaration = Declaration::parse_text(text)?;
            let exact_sources: Vec<(&str, &ObjectRef, Vec<String>)> = declaration
                .sources
                .iter()
                .map(|source| {
                    (
                        source.alias.as_str(),
                        &source.manifest,
                        selected_components(&declaration, &source.alias),
                    )
                })
                .collect();
            let stored_sources: Vec<(&str, &ObjectRef, Vec<String>)> = source_facts
                .iter()
                .map(|source| {
                    (
                        source.alias.as_str(),
                        &source.manifest,
                        source.components.clone(),
                    )
                })
                .collect();
            if exact_sources != stored_sources {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "derived source facts disagree with the locked declaration",
                );
            }
        } else if !source_facts.is_empty() {
            return refuse(
                Code::UNKNOWN_FIELD,
                "declaration-free abandonment cannot carry source facts",
            );
        }
        let mut row = TransactionRow {
            id,
            declaration,
            state,
            highest_session,
            current_session,
            added_parts,
            added_configs,
            source_facts,
            inherited_payload_objects,
            inherited_payload_bytes,
            header,
            manifest,
        };
        let before_parts = row.added_parts.clone();
        let before_configs = row.added_configs.clone();
        let before_sources = row.source_facts.clone();
        row.normalize();
        if row.added_parts != before_parts
            || row.added_configs != before_configs
            || row.source_facts != before_sources
        {
            return refuse(
                Code::SORT_ORDER,
                "derived additions must be strictly sorted and unique",
            );
        }
        if row.added_parts.windows(2).any(|w| {
            (&w[0].component, &w[0].key, &w[0].role) == (&w[1].component, &w[1].key, &w[1].role)
        }) || row.added_configs.windows(2).any(|w| w[0].name == w[1].name)
        {
            return refuse(Code::DUPLICATE_KEY, "duplicate derived addition record");
        }
        match row.state {
            TransactionState::Open => {
                if row.declaration.is_none()
                    || row.header.is_some()
                    || row.manifest.is_some()
                    || row.inherited_payload_objects != 0
                    || row.inherited_payload_bytes != 0
                {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        "open derived transaction has closed fields",
                    );
                }
            }
            TransactionState::Committed(_) => {
                if row.declaration.is_none()
                    || row.current_session.is_some()
                    || row.header.is_none()
                    || row.manifest.is_none()
                    || row.inherited_payload_objects > limits::MAX_TOTAL_REFS as u64
                    || row.inherited_payload_bytes > limits::MAX_TOTAL_BYTES
                {
                    return refuse(
                        Code::MISSING_FIELD,
                        "committed derived transaction is incomplete",
                    );
                }
            }
            TransactionState::Abandoned => {
                if row.current_session.is_some()
                    || row.header.is_some()
                    || row.manifest.is_some()
                    || !row.added_parts.is_empty()
                    || !row.added_configs.is_empty()
                    || row.inherited_payload_objects != 0
                    || row.inherited_payload_bytes != 0
                {
                    return refuse(
                        Code::UNKNOWN_FIELD,
                        "abandoned derived transaction has live fields",
                    );
                }
            }
        }
        Ok(row)
    }

    pub(crate) fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "added_configs".to_string(),
                Value::arr(
                    self.added_configs
                        .iter()
                        .map(AddedConfig::to_value)
                        .collect(),
                ),
            ),
            (
                "added_parts".to_string(),
                Value::arr(self.added_parts.iter().map(AddedPart::to_value).collect()),
            ),
            (
                "highest_session".to_string(),
                Value::uint(self.highest_session),
            ),
            ("id".to_string(), Value::str(self.id.clone())),
            (
                "inherited_payload_bytes".to_string(),
                Value::uint(self.inherited_payload_bytes),
            ),
            (
                "inherited_payload_objects".to_string(),
                Value::uint(self.inherited_payload_objects),
            ),
            (
                "source_facts".to_string(),
                Value::arr(
                    self.source_facts
                        .iter()
                        .map(|source| {
                            Value::obj(vec![
                                ("alias", Value::str(source.alias.clone())),
                                (
                                    "components",
                                    Value::arr(
                                        source
                                            .components
                                            .iter()
                                            .map(|component| Value::str(component.clone()))
                                            .collect(),
                                    ),
                                ),
                                ("header", source.header.to_value()),
                                ("manifest", source.manifest.to_value()),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("state".to_string(), Value::str(self.state.name())),
        ];
        if let Some(session) = self.current_session {
            fields.push(("current_session".to_string(), Value::uint(session)));
        }
        if let Some(declaration) = &self.declaration {
            fields.push(("declaration".to_string(), Value::str(declaration.clone())));
        }
        if let Some(header) = &self.header {
            fields.push(("header".to_string(), header.to_value()));
        }
        if let Some(manifest) = &self.manifest {
            fields.push(("manifest".to_string(), manifest.to_value()));
        }
        if let TransactionState::Committed(disposition) = &self.state {
            let value = disposition.to_value();
            fields.push(("disposition".to_string(), value));
        }
        Value::map(fields)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFact {
    pub alias: String,
    pub manifest: ObjectRef,
    pub header: ObjectRef,
    pub components: Vec<String>,
}

#[derive(Debug)]
struct SourceData {
    fact: SourceFact,
    manifest: Manifest,
    header: Header,
}

fn source<'a>(sources: &'a [SourceData], alias: &str) -> Result<&'a SourceData> {
    sources
        .iter()
        .find(|source| source.fact.alias == alias)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no source alias {alias:?} in the locked declaration"),
        })
}

fn require_payload_record(store: &Store, object: &ObjectRef, what: &str) -> Result<()> {
    match store.record_valid(&object.sha256) {
        Ok(record) if record.length == object.length => Ok(()),
        Ok(record) => refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{what}: {} is {} bytes in its verification record, reference says {}",
                object.id(),
                record.length,
                object.length
            ),
        ),
        Err(why) => refuse(
            Code::OBJECT_CORRUPT,
            format!(
                "{what}: {} has no current exact verification record ({why}); inherited payloads are carried by reference and never silently re-hashed",
                object.id()
            ),
        ),
    }
}

fn require_source_roots(store: &Store, declaration: &Declaration) -> Result<()> {
    let wanted: Vec<ObjectRef> = declaration
        .sources
        .iter()
        .map(|source| source.manifest.clone())
        .collect();
    match crate::gc::unretained_manifests(store, &wanted)?.first() {
        Some(missing) => refuse(
            Code::ROOT_ABSENT,
            format!(
                "source {} has no exact retained filesystem root",
                missing.id()
            ),
        ),
        None => Ok(()),
    }
}

fn load_sources(store: &Store, declaration: &Declaration) -> Result<Vec<SourceData>> {
    // A prepared model whose output a live source operation's publication custodian holds
    // derives by reference without its bytes; the result is publishable, never readable.
    let custodied = crate::ingest::custody::live(store.root())?;
    let mut out = Vec::new();
    for declared in &declaration.sources {
        let manifest = checkpoint::load_manifest(store, &declared.manifest)?;
        if manifest.manifest_id() != declared.manifest.id() {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                format!(
                    "source alias {:?}: manifest identity does not equal {}",
                    declared.alias,
                    declared.manifest.id()
                ),
            );
        }
        let header_ref = manifest.header().cloned().ok_or_else(|| Refusal {
            code: Code::ATTACHMENT_CARDINALITY,
            detail: format!(
                "source {} carries no CozyTensors header",
                declared.manifest.id()
            ),
        })?;
        let header = checkpoint::load_header(store, &header_ref)?;
        let closure = checkpoint::load_closure(store, &header)?;
        header.validate(&closure)?;
        let walk = checkpoint::walk_cozytensors(store, &manifest)?;
        walk.require_held(store, &custodied)?;
        // This is the zero-read/zero-hash inherit gate: the verification record proves the
        // exact installed inode/length/generation without opening or hashing payload bytes.
        for reached in &walk.objects {
            if (reached.kind == "part" || reached.kind == "config")
                && !custodied.contains(&reached.obj.sha256)
            {
                require_payload_record(store, &reached.obj, reached.kind)?;
            }
        }
        let components = selected_components(declaration, &declared.alias);
        out.push(SourceData {
            fact: SourceFact {
                alias: declared.alias.clone(),
                manifest: declared.manifest.clone(),
                header: header_ref,
                components,
            },
            manifest,
            header,
        });
    }
    Ok(out)
}

fn planned_part(declaration: &PartDeclaration) -> Result<Part> {
    let bytes = checked_bytes(
        "planned derived part",
        &declaration.shape,
        declaration.dtype,
    )?;
    let body = if Part::is_inline(bytes) {
        Body::Inline(vec![0; bytes as usize])
    } else {
        let mut left = bytes;
        let mut segments = Vec::new();
        while left > 0 {
            let length = left.min(limits::GRID_BYTES);
            segments.push(ObjectRef {
                sha256: "0".repeat(64),
                length,
            });
            left -= length;
        }
        Body::Segments(segments)
    };
    Ok(Part {
        dtype: declaration.dtype,
        shape: declaration.shape.clone(),
        body,
    })
}

#[derive(Debug)]
struct RegistryAdoption {
    spec: EncodingSpec,
    vectors: EncodingVectors,
}

/// Resolve by artifact identity alone. Registry aliases deliberately never cross this
/// boundary: they are discovery/display names, while the declaration is locked to a digest.
fn registry_adoption(encoding: &str) -> Result<RegistryAdoption> {
    let mut matches: Vec<_> = registry::seeds()
        .into_iter()
        .filter(|seed| seed.spec.object_id() == encoding)
        .collect();
    if matches.len() > 1 {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            format!(
                "target encoding {encoding} resolves to {} compiled registry entries",
                matches.len()
            ),
        );
    }
    let seed = matches.pop().ok_or_else(|| Refusal {
        code: Code::UNKNOWN_ENCODING,
        detail: format!(
            "target cites {encoding}, absent from every exact source closure and the compiled platform registry"
        ),
    })?;
    let spec = EncodingSpec::decode_nested(&seed.spec.canonical_bytes())?;
    let vectors = seed.vectors.ok_or_else(|| Refusal {
        code: Code::VECTORS_REQUIRED,
        detail: format!(
            "compiled registry spec {encoding} has no conformance-vector document and cannot be introduced into a derived manifest"
        ),
    })?;
    let vectors = EncodingVectors::parse_fixture(&vectors.fixture_bytes())?;
    let expected = spec.vectors.as_ref().ok_or_else(|| Refusal {
        code: Code::VECTOR_MISMATCH,
        detail: format!("compiled registry spec {encoding} does not reference its vectors"),
    })?;
    let actual = vectors.fixture_ref();
    if &actual != expected {
        return refuse(
            Code::VECTOR_MISMATCH,
            format!(
                "compiled registry spec {encoding} references {}, canonical vectors are {}",
                expected.id(),
                actual.id()
            ),
        );
    }
    vectors.check(&spec)?;
    Ok(RegistryAdoption { spec, vectors })
}

fn admit_registry(store: &Store, adoptions: &[RegistryAdoption]) -> Result<()> {
    for adoption in adoptions {
        // Qualification vectors remain fixture blobs. The encoding value itself is nested
        // in the CozyTensors header and is never admitted as another stored document.
        let _ = store;
        adoption.vectors.check(&adoption.spec)?;
    }
    Ok(())
}

fn graft_part(
    sources: &[SourceData],
    target: &TensorDeclaration,
    part: &PartDeclaration,
    selected: &PartSource,
) -> Result<Part> {
    let tensor = source(sources, &selected.source)?
        .header
        .components
        .iter()
        .find(|(component, _)| component == &selected.component)
        .ok_or_else(|| Refusal {
            code: Code::MISSING_COMPONENT,
            detail: "grafted source component is absent".into(),
        })?
        .1
        .iter()
        .find(|(key, _)| key == &selected.tensor)
        .ok_or_else(|| Refusal {
            code: Code::MISSING_TENSOR,
            detail: "grafted source tensor is absent".into(),
        })?;
    let tensor = &tensor.1;
    if tensor.dtype != target.dtype
        || tensor.shape != target.shape
        || tensor.encoding != target.encoding
    {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "grafted tensor changes logical dtype, shape or encoding",
        );
    }
    let inherited = tensor
        .parts
        .iter()
        .find(|(role, _)| role == &selected.role)
        .ok_or_else(|| Refusal {
            code: Code::MISSING_FIELD,
            detail: "grafted source role is absent".into(),
        })?;
    if inherited.1.dtype != part.dtype || inherited.1.shape != part.shape {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "grafted part changes stored dtype or shape",
        );
    }
    Ok(inherited.1.clone())
}

fn build_header(
    declaration: &Declaration,
    sources: &[SourceData],
    added_parts: &[AddedPart],
    added_configs: &[AddedConfig],
    planned: bool,
) -> Result<(Header, Closure, Vec<RegistryAdoption>)> {
    let mut components = Vec::new();
    for component in &declaration.components {
        let mut tensors = match (&component.source, &component.source_component) {
            (Some(alias), Some(source_component)) => source(sources, alias)?
                .header
                .components
                .iter()
                .find(|(name, _)| name == source_component)
                .map(|(_, tensors)| tensors.clone())
                .ok_or_else(|| Refusal {
                    code: Code::MISSING_COMPONENT,
                    detail: format!(
                        "source alias {alias:?} has no component {source_component:?} selected for target {:?}",
                        component.target
                    ),
                })?,
            (None, None) => Vec::new(),
            _ => unreachable!("declaration validation enforces the pair"),
        };

        for drop in &component.drop {
            let before = tensors.len();
            tensors.retain(|(key, _)| key != drop);
            if tensors.len() == before {
                return refuse(
                    Code::MISSING_TENSOR,
                    format!(
                        "target {:?} drops {drop:?}, which does not exist in its selected source component",
                        component.target
                    ),
                );
            }
        }
        for tensor in &component.add {
            if tensors.iter().any(|(key, _)| key == &tensor.key) {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    format!(
                        "target {:?} adds retained tensor {:?}; drop it explicitly before replacing it",
                        component.target, tensor.key
                    ),
                );
            }
            let mut parts = Vec::new();
            for part_declaration in &tensor.parts {
                let supplied = added_parts.iter().find(|added| {
                    added.component == component.target
                        && added.key == tensor.key
                        && added.role == part_declaration.role
                });
                let part = match (&part_declaration.source, supplied) {
                    (Some(selected), None) => {
                        graft_part(sources, tensor, part_declaration, selected)?
                    }
                    (Some(_), Some(_)) => {
                        return refuse(
                            Code::TRANSACTION_CONFLICT,
                            "grafted part cannot contain supplied bytes",
                        )
                    }
                    (None, Some(added)) => added.part.clone(),
                    (None, None) if planned => planned_part(part_declaration)?,
                    (None, None) => {
                        return refuse(
                            Code::ARTIFACT_INCOMPLETE,
                            format!(
                                "missing added value {}/{}#{}",
                                component.target, tensor.key, part_declaration.role
                            ),
                        )
                    }
                };
                if part.dtype != part_declaration.dtype || part.shape != part_declaration.shape {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        format!(
                            "added value {}/{}#{} does not match its locked dtype/shape declaration",
                            component.target, tensor.key, part_declaration.role
                        ),
                    );
                }
                parts.push((part_declaration.role.clone(), part));
            }
            tensors.push((
                tensor.key.clone(),
                Tensor {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                    encoding: tensor.encoding.clone(),
                    parts,
                },
            ));
        }
        if tensors.is_empty() {
            return refuse(
                Code::EMPTY_COMPONENTS,
                format!("target component {:?} is empty", component.target),
            );
        }
        components.push((component.target.clone(), tensors));
    }

    let mut rows = std::collections::HashMap::new();
    for (component, tensors) in components.drain(..) {
        for (key, tensor) in tensors {
            if rows
                .insert((component.clone(), key.clone()), tensor)
                .is_some()
            {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("derived target repeats {component}/{key}"),
                );
            }
        }
    }
    let mut ordered: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    let mut closed_components = HashSet::new();
    for (component, key) in &declaration.order {
        let tensor = rows
            .remove(&(component.clone(), key.clone()))
            .ok_or_else(|| Refusal {
                code: Code::TRAVERSAL_INCOMPLETE,
                detail: format!("construction order repeats or invents {component}/{key}"),
            })?;
        match ordered.last_mut() {
            Some((current, tensors)) if current == component => {
                tensors.push((key.clone(), tensor));
            }
            _ => {
                if !closed_components.insert(component.clone()) {
                    return refuse(
                        Code::TRAVERSAL_ORDER_MISMATCH,
                        format!("construction order interleaves component {component:?}"),
                    );
                }
                ordered.push((component.clone(), vec![(key.clone(), tensor)]));
            }
        }
    }
    if !rows.is_empty() {
        let mut missing: Vec<String> = rows
            .keys()
            .map(|(component, key)| format!("{component}/{key}"))
            .collect();
        missing.sort();
        return refuse(
            Code::TRAVERSAL_INCOMPLETE,
            format!(
                "construction order omits {} derived tensor(s), first {:?}",
                missing.len(),
                &missing[..missing.len().min(3)]
            ),
        );
    }
    components = ordered;

    let mut configs = Vec::new();
    for config in &declaration.configs {
        match config {
            ConfigDeclaration::Copy {
                target,
                source: alias,
                source_config,
            } => {
                let value = source(sources, alias)?
                    .header
                    .configs
                    .iter()
                    .find(|(name, _)| name == source_config)
                    .map(|(_, value)| value.clone())
                    .ok_or_else(|| Refusal {
                        code: Code::MISSING_FIELD,
                        detail: format!(
                            "source alias {alias:?} has no config {source_config:?} selected for {target:?}"
                        ),
                    })?;
                configs.push((target.clone(), value));
            }
            ConfigDeclaration::Add { target } => {
                let value = if planned {
                    b"{}".to_vec()
                } else {
                    added_configs
                        .iter()
                        .find(|added| added.name == *target)
                        .map(|added| added.config.clone())
                        .ok_or_else(|| Refusal {
                            code: Code::ARTIFACT_INCOMPLETE,
                            detail: format!("missing added config {target:?}"),
                        })?
                };
                configs.push((target.clone(), value));
            }
            ConfigDeclaration::Derive { target, .. } => {
                let value = if planned {
                    b"{}".to_vec()
                } else {
                    added_configs
                        .iter()
                        .find(|added| added.name == *target)
                        .map(|added| added.config.clone())
                        .ok_or_else(|| Refusal {
                            code: Code::ARTIFACT_INCOMPLETE,
                            detail: format!("missing derived config {target:?}"),
                        })?
                };
                configs.push((target.clone(), value));
            }
        }
    }

    let cited: HashSet<String> = components
        .iter()
        .flat_map(|(_, tensors)| tensors.iter().map(|(_, tensor)| tensor.encoding.clone()))
        .collect();
    let mut encodings = Vec::new();
    let mut closure = Closure::default();
    let mut registry_adoptions = Vec::new();
    for encoding in cited {
        let mut found: Option<crate::spec::EncodingSpec> = None;
        for source in sources {
            if let Some(spec) = source
                .header
                .encodings
                .iter()
                .find(|spec| spec.object_id() == encoding)
                .cloned()
            {
                match &found {
                    Some(existing) if existing != &spec => {
                        return refuse(
                            Code::TRANSACTION_CONFLICT,
                            format!(
                                "encoding {encoding} has conflicting definitions across sources"
                            ),
                        )
                    }
                    None => found = Some(spec),
                    _ => {}
                }
            }
        }
        let spec = match found {
            Some(found) => found,
            None => {
                let adoption = registry_adoption(&encoding)?;
                let found = adoption.spec.clone();
                registry_adoptions.push(adoption);
                found
            }
        };
        encodings.push(spec.clone());
        closure.insert(spec);
    }
    encodings.sort_by_key(crate::spec::EncodingSpec::object_id);
    configs.sort_by(|a, b| a.0.cmp(&b.0));
    let header = Header {
        configs,
        assets: companion_files(declaration, sources)?
            .into_iter()
            .filter_map(|(path, entry)| match entry {
                Entry::File(object) if object.length != 0 => {
                    let asset = Asset {
                        logical_sha256: object.sha256.clone(),
                        logical_length: object.length,
                        media_type: "application/octet-stream".into(),
                        segments: vec![object],
                    };
                    Some((path, asset))
                }
                _ => None,
            })
            .collect(),
        encodings,
        components,
    };
    header.validate(&closure)?;
    Ok((header, closure, registry_adoptions))
}

pub struct Begin {
    pub writer_hold: Hold,
    pub source_leases: Vec<(String, ReadLease)>,
    pub source_views: Vec<SourceView>,
    pub additions: Additions,
}

#[derive(Debug, Clone)]
pub struct SourceView {
    pub fact: SourceFact,
    pub header: Header,
    pub configs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InspectedConfig {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SourceInspection {
    pub view: SourceView,
    pub configs: Vec<InspectedConfig>,
}

/// Bounded metadata/config preview for a source the caller already holds by exact manifest.
/// Config bytes are inline in the verified header; tensor payload ObjectRefs never leave TensorFS.
pub fn inspect_source(
    store: &Store,
    meta: &Meta,
    manifest: ObjectRef,
    component_names: Vec<String>,
    config_names: Vec<String>,
) -> Result<SourceInspection> {
    let hold = meta.acquire_hold("derived-source-inspect")?;
    let result = inspect_retained_source(store, manifest, component_names, config_names);
    let released = hold.release(meta);
    result.and_then(|inspection| released.map(|()| inspection))
}

fn inspect_retained_source(
    store: &Store,
    manifest: ObjectRef,
    component_names: Vec<String>,
    config_names: Vec<String>,
) -> Result<SourceInspection> {
    let alias = "source".to_string();
    let mut declaration = Declaration {
        files: Vec::new(),
        objects: Vec::new(),
        work_fingerprint: None,
        sources: vec![Source {
            alias: alias.clone(),
            manifest,
        }],
        components: component_names
            .iter()
            .map(|component| ComponentDeclaration {
                target: component.clone(),
                source: Some(alias.clone()),
                source_component: Some(component.clone()),
                drop: Vec::new(),
                add: Vec::new(),
            })
            .collect(),
        configs: config_names
            .iter()
            .map(|config| ConfigDeclaration::Copy {
                target: config.clone(),
                source: alias.clone(),
                source_config: config.clone(),
            })
            .collect(),
        order: Vec::new(),
        max_new_bytes: 0,
    };
    declaration.normalize_and_validate()?;
    require_source_roots(store, &declaration)?;
    let mut sources = load_sources(store, &declaration)?;
    let source = sources.pop().ok_or_else(|| Refusal {
        code: Code::ROOT_ABSENT,
        detail: "source inspection resolved no source".into(),
    })?;
    for component in &component_names {
        if !source
            .header
            .components
            .iter()
            .any(|(name, _)| name == component)
        {
            return refuse(
                Code::MISSING_FIELD,
                format!("source has no component {component:?}"),
            );
        }
    }
    let configs = config_names
        .iter()
        .map(|name| {
            source
                .header
                .configs
                .iter()
                .find(|(candidate, _)| candidate == name)
                .map(|(_, config)| InspectedConfig {
                    name: name.clone(),
                    bytes: config.clone(),
                })
                .ok_or_else(|| Refusal {
                    code: Code::MISSING_FIELD,
                    detail: format!("source has no config {name:?}"),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SourceInspection {
        view: SourceView {
            fact: source.fact.clone(),
            header: source.header.clone(),
            configs: config_names,
        },
        configs,
    })
}

fn release_guards(meta: &Meta, writer_hold: Option<Hold>, source_leases: Vec<(String, ReadLease)>) {
    for (_, lease) in source_leases {
        let _ = lease.release(meta);
    }
    if let Some(hold) = writer_hold {
        let _ = hold.release(meta);
    }
}

// One epoch/declaration decision for inspection and writer creation. The returned
// row is an isolated candidate; only begin publishes it under the native CAS.
fn planned_row(
    initial: Option<&TransactionRow>,
    transaction: &str,
    writer_session: u64,
    declaration_text: &str,
    source_facts: &[SourceFact],
) -> Result<TransactionRow> {
    if let Some(previous) = initial {
        match previous.state {
            TransactionState::Committed(_) => {
                return refuse(
                    Code::TRANSACTION_CLOSED,
                    format!("{transaction} is committed; recover it with derived_lookup"),
                )
            }
            TransactionState::Abandoned => {
                return refuse(
                    Code::TRANSACTION_CLOSED,
                    format!("{transaction} was durably abandoned and cannot reopen"),
                )
            }
            TransactionState::Open => {}
        }
        if previous.declaration.as_deref() != Some(declaration_text)
            || previous.source_facts.as_slice() != source_facts
        {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                format!("{transaction} is already locked to a different declaration"),
            );
        }
        if writer_session < previous.highest_session
            || (writer_session == previous.highest_session
                && previous.current_session != Some(writer_session))
        {
            return refuse(
                Code::WRITER_FENCED,
                format!(
                    "WriterSessionId {writer_session} is behind the durable fence {}",
                    previous.highest_session
                ),
            );
        }
        let mut row = previous.clone();
        row.highest_session = row.highest_session.max(writer_session);
        row.current_session = Some(writer_session);
        return Ok(row);
    }
    Ok(TransactionRow {
        id: transaction.to_string(),
        declaration: Some(declaration_text.to_string()),
        state: TransactionState::Open,
        highest_session: writer_session,
        current_session: Some(writer_session),
        added_parts: Vec::new(),
        added_configs: Vec::new(),
        source_facts: source_facts.to_vec(),
        inherited_payload_objects: 0,
        inherited_payload_bytes: 0,
        header: None,
        manifest: None,
    })
}

/// The Store a live writer admits through. On a persistent disk its parts and progress
/// documents are not synced one by one: each checkpoint (and the commit) makes everything
/// admitted since the previous one durable at once, before the root names it, so a crash
/// loses at most the work after the last checkpoint. An ephemeral disk needs no sync at all.
pub fn writer_store(store: &Store) -> Result<Store> {
    Ok(store.syncing(crate::unsynced::Epoch::begin_every(store, u64::MAX)?))
}

/// The writer is done: sync anything admitted since its last checkpoint and retire its epoch.
pub fn finish_writer(store: &Store) -> Result<()> {
    store.finish_admitted()
}

pub fn begin(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    mut declaration: Declaration,
    checkpoint: Option<&ObjectRef>,
) -> Result<Begin> {
    let transaction = transaction_id(transaction)?;
    if writer_session == 0 || writer_session > limits::INT_MAX as u64 {
        return refuse(
            Code::NUMBER_RANGE,
            "WriterSessionId must be positive and inside the canonical integer range",
        );
    }
    let declaration_text =
        String::from_utf8(declaration.canonical_work_bytes()?).expect("canonical JSON is UTF-8");

    // The writer hold is registered before source metadata is touched. It is the live
    // process proof that keeps exclusive GC out even for a source-free create.
    let writer_hold = meta.acquire_hold("derived-writer")?;
    let sources = match require_source_roots(store, &declaration)
        .and_then(|()| load_sources(store, &declaration))
    {
        Ok(sources) => sources,
        Err(error) => {
            release_guards(meta, Some(writer_hold), Vec::new());
            return Err(error);
        }
    };
    if let Err(error) = build_header(&declaration, &sources, &[], &[], true) {
        release_guards(meta, Some(writer_hold), Vec::new());
        return Err(error);
    }
    // Pin every resident source object. A custodied object has no local bytes to pin: a
    // derivation over it carries it by reference and never reads it.
    let custodied = match crate::ingest::custody::live(store.root()) {
        Ok(custodied) => custodied,
        Err(error) => {
            release_guards(meta, Some(writer_hold), Vec::new());
            return Err(error);
        }
    };
    let mut source_leases = Vec::new();
    for source in &sources {
        let pinned = if custodied.is_empty() {
            read::acquire_cozytensors(store, meta, &source.manifest)
        } else {
            checkpoint::walk_cozytensors(store, &source.manifest).and_then(|walk| {
                let id = source.manifest.manifest_id();
                read::acquire(
                    store,
                    meta,
                    id.strip_prefix("sha256:").unwrap_or(&id),
                    walk.distinct()
                        .into_iter()
                        .filter(|object| !custodied.contains(&object.sha256))
                        .cloned()
                        .collect(),
                )
            })
        };
        match pinned {
            Ok((lease, receipts)) if receipts.0.is_empty() => {
                source_leases.push((source.fact.alias.clone(), lease));
            }
            Ok((lease, receipts)) => {
                let _ = lease.release(meta);
                release_guards(meta, Some(writer_hold), source_leases);
                return refuse(
                    Code::OBJECT_CORRUPT,
                    format!(
                        "source {:?} required {} rehash receipt(s) while acquiring its pinned read lease; an inherited payload must carry from a current verification record",
                        source.fact.alias,
                        receipts.0.len()
                    ),
                );
            }
            Err(error) => {
                release_guards(meta, Some(writer_hold), source_leases);
                return Err(error);
            }
        }
    }
    let source_facts: Vec<SourceFact> = sources.iter().map(|source| source.fact.clone()).collect();
    let source_views: Vec<SourceView> = sources
        .iter()
        .map(|source| SourceView {
            fact: source.fact.clone(),
            header: source.header.clone(),
            configs: selected_configs(&declaration, &source.fact.alias),
        })
        .collect();

    // A checkpoint this work cannot continue (another plan or work context, or no longer
    // whole) is discarded: the work re-derives what it lost instead of refusing.
    let mut checkpoint = checkpoint;
    let incoming = match checkpoint
        .map(|head| read_progress(store, head, &transaction, &declaration_text, &source_facts))
    {
        None => None,
        Some(Ok(row)) => Some(row),
        Some(Err(error)) if error.code == Code::IO_FAILED => {
            release_guards(meta, Some(writer_hold), source_leases);
            return Err(error);
        }
        Some(Err(_)) => {
            checkpoint = None;
            None
        }
    };
    // Byte verification can update the verification catalog. Keep it outside the SQL
    // writer transaction, then compare the observed row before assigning this epoch.
    let initial = meta.derived_row(&transaction)?;
    let mut row = match planned_row(
        initial.as_ref(),
        &transaction,
        writer_session,
        &declaration_text,
        &source_facts,
    ) {
        Ok(row) => row,
        Err(error) => {
            release_guards(meta, Some(writer_hold), source_leases);
            return Err(error);
        }
    };
    let merged = merge_progress(store, &mut row, incoming.as_ref()).and_then(|()| {
        build_header(
            &declaration,
            &sources,
            &row.added_parts,
            &row.added_configs,
            true,
        )?;
        Ok(())
    });
    if let Err(error) = merged {
        release_guards(meta, Some(writer_hold), source_leases);
        return Err(error);
    }
    let committed = (|| {
        let mut txn = meta.txn_for(&transaction)?;
        let current = txn.rows.iter_mut().find(|row| row.id == transaction);
        if current.as_deref() != initial.as_ref() {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "derived work changed while beginning; retry with current writer epoch",
            );
        }
        if roots::read(store, &transaction)?
            .and_then(|root| root.restore_epoch)
            .is_some_and(|epoch| epoch > writer_session)
        {
            return refuse(
                Code::WRITER_FENCED,
                "writer is behind the retained restore epoch",
            );
        }
        roots::rewrite_parts(store, &transaction, &added_objects(&row))?;
        retain_progress(store, &row, checkpoint)?;
        match current {
            Some(current) => *current = row.clone(),
            None => txn.rows.push(row.clone()),
        }
        txn.commit()
    })();
    if let Err(error) = committed {
        release_guards(meta, Some(writer_hold), source_leases);
        return Err(error);
    }
    Ok(Begin {
        writer_hold,
        source_leases,
        source_views,
        additions: Additions::of(&declaration),
    })
}

fn progress_row(row: &TransactionRow) -> TransactionRow {
    let mut progress = row.clone();
    progress.highest_session = 0;
    progress.current_session = None;
    progress
}

fn progress_plan(row: &TransactionRow) -> String {
    let mut identity = progress_row(row);
    identity.added_parts.clear();
    identity.added_configs.clear();
    let mut hash = crate::sha256::Sha256::new();
    hash.update(b"tensorfs.derived.work/1\0");
    hash.update(&crate::canon::write(&identity.to_value()));
    format!("sha256:{}", crate::sha256::hex(&hash.finish()))
}

fn validate_additions(row: &TransactionRow) -> Result<()> {
    let declaration = declaration(row)?;
    for added in &row.added_parts {
        let expected = declaration
            .components
            .iter()
            .find(|c| c.target == added.component)
            .and_then(|c| c.add.iter().find(|t| t.key == added.key))
            .and_then(|t| t.parts.iter().find(|p| p.role == added.role));
        if !expected.is_some_and(|expected| {
            expected.source.is_none()
                && expected.dtype == added.part.dtype
                && expected.shape == added.part.shape
        }) {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "progress contains an undeclared or changed tensor part",
            );
        }
    }
    for added in &row.added_configs {
        if !declaration.configs.iter().any(|config| matches!(config,
            ConfigDeclaration::Add { target } | ConfigDeclaration::Derive { target, .. } if target == &added.name)) {
            return refuse(Code::TRANSACTION_CONFLICT, "progress contains an undeclared config");
        }
    }
    Ok(())
}

fn merge_progress(
    store: &Store,
    row: &mut TransactionRow,
    incoming: Option<&TransactionRow>,
) -> Result<()> {
    validate_additions(row)?;
    row.added_parts
        .retain(|added| crate::ingest::journal::segments_still_stand(store, &added.part));
    if let Some(incoming) = incoming {
        validate_additions(incoming)?;
        for added in &incoming.added_parts {
            if !crate::ingest::journal::segments_still_stand(store, &added.part) {
                continue;
            }
            match row.added_parts.iter().find(|old| {
                old.component == added.component && old.key == added.key && old.role == added.role
            }) {
                Some(old) if old != added => {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "accepted tensor part conflicts with restored progress",
                    )
                }
                Some(_) => {}
                None => row.added_parts.push(added.clone()),
            }
        }
        for added in &incoming.added_configs {
            match row.added_configs.iter().find(|old| old.name == added.name) {
                Some(old) if old != added => {
                    return refuse(
                        Code::TRANSACTION_CONFLICT,
                        "accepted config conflicts with restored progress",
                    )
                }
                Some(_) => {}
                None => row.added_configs.push(added.clone()),
            }
        }
    }
    row.normalize();
    Ok(())
}

/// Verify a restored checkpoint against the current native declaration without creating,
/// importing or fencing writer authority. Payload admission must already have completed.
#[allow(clippy::too_many_arguments)]
pub fn validate_checkpoint(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    declaration_bytes: &[u8],
    head: &ObjectRef,
    operation: &str,
    slot: &str,
    plan: &str,
) -> Result<()> {
    let hold = meta.acquire_hold("derived-checkpoint-inspect")?;
    let result = validate_retained_checkpoint(
        store,
        transaction,
        declaration_bytes,
        head,
        operation,
        slot,
        plan,
    );
    let released = hold.release(meta);
    result.and(released)
}

fn validate_retained_checkpoint(
    store: &Store,
    transaction: &str,
    declaration_bytes: &[u8],
    head: &ObjectRef,
    operation: &str,
    slot: &str,
    plan: &str,
) -> Result<()> {
    let transaction = transaction_id(transaction)?;
    let mut declaration = Declaration::from_value(&crate::canon::parse_canonical(
        declaration_bytes,
        limits::DOC_MAX_BYTES / 2,
    )?)?;
    let canonical = declaration.canonical_work_bytes()?;
    if canonical != declaration_bytes {
        return refuse(
            Code::NONCANONICAL_ENCODING,
            "derived declaration differs from its canonical bytes",
        );
    }
    let link = crate::durability::local_link(store, head)?;
    if link.operation != format!("{operation}/{slot}") || link.plan != plan {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "derived checkpoint differs from current operation, slot or plan",
        );
    }
    require_source_roots(store, &declaration)?;
    let sources = load_sources(store, &declaration)?;
    let facts = sources
        .iter()
        .map(|source| source.fact.clone())
        .collect::<Vec<_>>();
    let text = std::str::from_utf8(&canonical).expect("canonical JSON is UTF-8");
    let row = read_progress(store, head, &transaction, text, &facts)?;
    validate_additions(&row)?;
    build_header(
        &declaration,
        &sources,
        &row.added_parts,
        &row.added_configs,
        true,
    )?;
    let chain = crate::durability::local_chain(store, head, plan)?;
    for object in chain.blobs.iter().chain(added_objects(&row).iter()) {
        require_payload_record(store, object, "restored derived checkpoint")?;
    }
    Ok(())
}

// Closed facade result: part keys and config names, with no mutable journal access.
#[allow(clippy::type_complexity)]
pub fn completed(
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
) -> Result<(Vec<(String, String, String)>, Vec<String>)> {
    let mut rows: Vec<_> = meta.derived_row(transaction)?.into_iter().collect();
    let row = open_row(&mut rows, transaction, writer_session)?;
    Ok((
        row.added_parts
            .iter()
            .map(|p| (p.component.clone(), p.key.clone(), p.role.clone()))
            .collect(),
        row.added_configs.iter().map(|c| c.name.clone()).collect(),
    ))
}

/// A writer's position in its checkpoint chain: the chain's tip and held blobs, its fixed
/// plan and work context, and the parts it accepted since. With it a checkpoint writes and
/// verifies only those parts instead of re-reading the chain it extends.
#[derive(Debug, Clone)]
pub struct CheckpointCursor {
    chain: String,
    plan: String,
    work: ObjectRef,
    tip: crate::durability::Tip,
    parts: usize,
    pending: Vec<AddedPart>,
}

impl CheckpointCursor {
    /// Seed from a checkpoint the full path just wrote or confirmed. `None` when there is
    /// no head yet to extend.
    fn at(
        store: &Store,
        meta: &Meta,
        transaction: &str,
        writer_session: u64,
        chain: String,
        plan: &str,
        head: &crate::durability::Head,
    ) -> Result<Option<Self>> {
        let Some(tip) = head.head.as_ref() else {
            return Ok(None);
        };
        let mut rows: Vec<_> = meta.derived_row(transaction)?.into_iter().collect();
        let row = open_row(&mut rows, transaction, writer_session)?;
        let context = progress::context(row);
        let held = crate::durability::local_chain(store, tip, plan)?
            .blobs
            .into_iter()
            .map(|object| object.sha256)
            .collect();
        Ok(Some(Self {
            chain,
            plan: plan.to_string(),
            work: ObjectRef::of(&crate::canon::write(&context.to_value())),
            tip: crate::durability::Tip {
                head: head.clone(),
                held,
            },
            parts: row.added_parts.len(),
            pending: Vec::new(),
        }))
    }

    /// Extend the chain by the pending parts, or `None` when anything but this writer's own
    /// accepted parts moved since the cursor was taken.
    fn advance(
        &mut self,
        store: &Store,
        meta: &Meta,
        transaction: &str,
        writer_session: u64,
        chain: &str,
        previous: Option<&ObjectRef>,
    ) -> Result<Option<(String, crate::durability::Head)>> {
        let tip = self.tip.head.head.clone();
        let root = roots::read(store, transaction)?;
        if self.chain != chain
            || previous.is_some_and(|previous| Some(previous) != tip.as_ref())
            || root.as_ref().and_then(|root| root.checkpoint.as_ref()) != tip.as_ref()
        {
            return Ok(None);
        }
        meta.check_session(transaction, writer_session)?;
        if meta.part_count(transaction)? != self.parts + self.pending.len() {
            return Ok(None);
        }
        if self.pending.is_empty() {
            return Ok(Some((self.plan.clone(), self.tip.head.clone())));
        }
        let parts = std::mem::take(&mut self.pending);
        let objects = parts
            .iter()
            .flat_map(|added| added.part.segments().iter().cloned())
            .collect();
        let count = parts.len();
        let delta = progress::Delta::accepted(self.work.clone(), parts);
        let policy = crate::durability::Policy {
            chain,
            operation: transaction,
            plan: &self.plan,
            interval: crate::durability::INTERVAL_BYTES,
        };
        let extended = crate::durability::extend(
            store,
            &policy,
            &self.tip,
            &crate::canon::write(&delta.to_value()),
            objects,
            None,
        )?;
        let Some(mut root) = root else {
            return Ok(None);
        };
        // Everything the new head names is durable before the root publishes it.
        store.sync_admitted()?;
        root.checkpoint = extended.head.head.clone();
        roots::write(store, &root)?;
        self.tip
            .held
            .extend(extended.named.into_iter().map(|object| object.sha256));
        self.tip.head = extended.head;
        self.parts += count;
        Ok(Some((self.plan.clone(), self.tip.head.clone())))
    }
}

/// Checkpoint through the writer's cursor, which each full checkpoint re-seeds.
#[allow(clippy::too_many_arguments)]
pub fn checkpoint_indexed(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    operation: &str,
    slot: &str,
    previous: Option<&ObjectRef>,
    cursor: &mut Option<CheckpointCursor>,
) -> Result<(String, crate::durability::Head)> {
    let chain = format!("{operation}/{slot}");
    if let Some(current) = cursor.as_mut() {
        if let Some(advanced) =
            current.advance(store, meta, transaction, writer_session, &chain, previous)?
        {
            return Ok(advanced);
        }
    }
    *cursor = None;
    let (plan, head) = checkpoint_progress(
        store,
        meta,
        transaction,
        writer_session,
        operation,
        slot,
        previous,
    )?;
    *cursor = CheckpointCursor::at(
        store,
        meta,
        transaction,
        writer_session,
        chain,
        &plan,
        &head,
    )?;
    Ok((plan, head))
}

pub fn checkpoint_progress(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    operation: &str,
    slot: &str,
    previous: Option<&ObjectRef>,
) -> Result<(String, crate::durability::Head)> {
    let progress = {
        let mut txn = meta.txn_for(transaction)?;
        progress_row(open_row(&mut txn.rows, transaction, writer_session)?)
    };
    let plan = progress_plan(&progress);
    let chain = format!("{operation}/{slot}");
    ascii_name("checkpoint operation", &chain, limits::MAX_NAME_BYTES)?;
    let policy = crate::durability::Policy {
        chain: &chain,
        operation: transaction,
        plan: &plan,
        interval: crate::durability::INTERVAL_BYTES,
    };
    let root = roots::read(store, transaction)?;
    let previous = crate::durability::checkpoint_predecessor(
        store,
        &policy,
        root.as_ref().and_then(|root| root.checkpoint.as_ref()),
        previous,
    )?;
    let work = progress::context(&progress);
    let work_bytes = crate::canon::write(&work.to_value());
    let work_ref = ObjectRef::of(&work_bytes);
    let prior = previous
        .as_ref()
        .map(|head| {
            read_progress(
                store,
                head,
                transaction,
                progress.declaration.as_deref().expect("open declaration"),
                &progress.source_facts,
            )
        })
        .transpose()?;
    let delta = progress::Delta::between(&progress, prior.as_ref(), work_ref.clone())?;
    if delta.is_empty() && previous.is_some() {
        let prior = crate::durability::local_chain(store, previous.as_ref().unwrap(), &plan)?;
        return Ok((
            plan,
            crate::durability::Head {
                head: previous,
                links: prior.links,
                bytes: prior.bytes,
            },
        ));
    }
    store.put_stream(
        &mut work_bytes.as_slice(),
        Some(&work_ref),
        &Default::default(),
    )?;
    let bytes = crate::canon::write(&delta.to_value());
    let mut objects = added_objects(&progress);
    objects.push(work_ref);
    let exported = crate::durability::checkpoint_bytes(
        store,
        &policy,
        previous.as_ref(),
        &bytes,
        objects,
        None,
    )?;
    store.sync_admitted()?;
    let mut txn = meta.txn_for(transaction)?;
    let row = open_row(&mut txn.rows, transaction, writer_session)?;
    retain_progress(store, row, exported.head.head.as_ref())?;
    Ok((plan, exported.head))
}

/// Private filesystem roots are the only GC liveness authority for accepted work.
fn retain_progress(
    store: &Store,
    row: &TransactionRow,
    checkpoint: Option<&ObjectRef>,
) -> Result<()> {
    let previous = roots::read(store, &row.id)?;
    let local = previous.as_ref().and_then(|root| root.checkpoint.as_ref());
    let head = if let Some(supplied) = checkpoint {
        let link = crate::durability::local_link(store, supplied)?;
        let plan = progress_plan(row);
        crate::durability::checkpoint_predecessor(
            store,
            &crate::durability::Policy {
                chain: &link.operation,
                operation: &row.id,
                plan: &plan,
                interval: crate::durability::INTERVAL_BYTES,
            },
            local,
            Some(supplied),
        )?
    } else {
        local.cloned()
    };
    roots::write(
        store,
        &roots::Root {
            tensorfs: Some(crate::VERSION.into()),
            transaction: row.id.clone(),
            manifest: None,
            name: None,
            // Accepted parts are named by the append-only parts journal.
            blobs: Vec::new(),
            sources: row
                .source_facts
                .iter()
                .map(|source| source.manifest.clone())
                .collect(),
            checkpoint: head,
            restore_epoch: None,
        },
    )
}

/// Durability calls this under a live GC guard, before fetch and after verified admission.
/// The same SQL lock as abandonment prevents a late download recreating a disposed root.
pub(crate) fn retain_restore(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    epoch: u64,
    object: Option<&ObjectRef>,
) -> Result<()> {
    let transaction = transaction_id(transaction)?;
    if epoch == 0 || epoch > limits::INT_MAX as u64 {
        return refuse(
            Code::NUMBER_RANGE,
            "restore writer epoch is outside the canonical range",
        );
    }
    if let Some(object) = object {
        require_payload_record(store, object, "restored checkpoint object")?;
    }
    let txn = meta.txn_for(&transaction)?;
    let row = txn.rows.iter().find(|row| row.id == transaction);
    let active = if let Some(row) = row {
        if row.state != TransactionState::Open {
            return refuse(
                Code::TRANSACTION_CLOSED,
                "restored transaction is already closed",
            );
        }
        if epoch < row.highest_session
            || epoch == row.highest_session && row.current_session != Some(epoch)
        {
            return refuse(
                Code::WRITER_FENCED,
                "restored object is behind the writer fence",
            );
        }
        if row.current_session.is_some_and(|writer| writer != epoch) {
            return refuse(
                Code::STORE_BUSY,
                "prior writer must be fenced before restoration",
            );
        }
        row.current_session == Some(epoch)
    } else {
        false
    };
    let mut root = roots::read(store, &transaction)?.unwrap_or_else(|| roots::Root {
        tensorfs: Some(crate::VERSION.into()),
        transaction: transaction.clone(),
        manifest: None,
        blobs: Vec::new(),
        sources: Vec::new(),
        checkpoint: None,
        name: None,
        restore_epoch: None,
    });
    if root.restore_epoch.is_some_and(|held| held > epoch) {
        return refuse(
            Code::WRITER_FENCED,
            "restored object is behind the retained restore epoch",
        );
    }
    if !active {
        root.restore_epoch = Some(epoch);
    }
    if let Some(object) = object {
        if let Some(held) = root.blobs.iter().find(|held| held.sha256 == object.sha256) {
            if held != object {
                return refuse(Code::LENGTH_MISMATCH, "restore hold changed object length");
            }
        } else {
            root.blobs.push(object.clone());
        }
        root.blobs
            .sort_by(|left, right| left.sha256.cmp(&right.sha256));
    }
    roots::write(store, &root)
}

/// The manifests private roots name themselves, without walking their checkpoint chains.
pub(crate) fn rooted_manifests(store: &Store) -> Result<Vec<ObjectRef>> {
    let mut named: Vec<ObjectRef> = retention::held_objects(store)?
        .into_iter()
        .map(|hold| ObjectRef {
            sha256: hold.sha256,
            length: hold.length,
        })
        .collect();
    for root in roots::all(store)? {
        named.extend(root.sources);
        named.extend(root.manifest);
    }
    Ok(named)
}

pub(crate) fn retained_objects(store: &Store) -> Result<Vec<crate::storage::HeldKey>> {
    let mut holds = retention::held_objects(store)?;
    for object in roots::journaled(store)? {
        holds.push(crate::storage::HeldKey {
            key: crate::storage::blob_key(&object.sha256)?,
            kind: "blob".into(),
            length: object.length,
            sha256: object.sha256,
        });
    }
    for root in roots::all(store)? {
        let mut blobs = root.blobs;
        let mut manifests = root.sources;
        manifests.extend(root.manifest);
        if let Some(head) = root.checkpoint {
            let link = crate::durability::local_link(store, &head)?;
            let chain = crate::durability::local_chain(store, &head, &link.plan)?;
            blobs.extend(chain.blobs);
            blobs.extend(crate::durability::local_documents(store, &head)?);
            manifests.extend(chain.manifests);
        }
        for (kind, objects) in [("blob", blobs), ("manifest", manifests)] {
            for object in objects {
                let key = if kind == "blob" {
                    crate::storage::blob_key(&object.sha256)?
                } else {
                    crate::storage::manifest_key(&object.sha256)?
                };
                holds.push(crate::storage::HeldKey {
                    key,
                    kind: kind.into(),
                    length: object.length,
                    sha256: object.sha256,
                });
            }
        }
    }
    Ok(holds)
}

pub(crate) fn open_row<'a>(
    rows: &'a mut [TransactionRow],
    transaction: &str,
    writer_session: u64,
) -> Result<&'a mut TransactionRow> {
    let row = rows
        .iter_mut()
        .find(|row| row.id == transaction)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no derived transaction {transaction}"),
        })?;
    if row.state != TransactionState::Open {
        return refuse(
            Code::TRANSACTION_CLOSED,
            format!("{transaction} is {}", row.state.name()),
        );
    }
    if row.current_session != Some(writer_session) {
        return refuse(
            Code::WRITER_FENCED,
            format!(
                "WriterSessionId {writer_session} is not current for {transaction}; current={:?}, fence={}",
                row.current_session, row.highest_session
            ),
        );
    }
    Ok(row)
}

pub fn check_writer(meta: &Meta, transaction: &str, writer_session: u64) -> Result<()> {
    let transaction = transaction_id(transaction)?;
    let mut rows: Vec<_> = meta.derived_row(&transaction)?.into_iter().collect();
    open_row(&mut rows, &transaction, writer_session)?;
    Ok(())
}

fn declaration(row: &TransactionRow) -> Result<Declaration> {
    Declaration::parse_text(row.declaration.as_deref().ok_or_else(|| Refusal {
        code: Code::MISSING_FIELD,
        detail: format!("transaction {} has no declaration", row.id),
    })?)
}

/// The exact native payload limit, checked against the current writer before
/// an embedding owner admits storage or asks a producer to supply any bytes.
/// One writer session's parts to write, parsed once from its immutable declaration. A
/// part costs one lookup here instead of a parse of the whole declaration.
#[derive(Debug, Clone, Default)]
pub struct Additions {
    parts: std::collections::HashMap<(String, String, String), PartDeclaration>,
}

impl Additions {
    pub fn of(declaration: &Declaration) -> Self {
        let mut parts = std::collections::HashMap::new();
        for component in &declaration.components {
            for tensor in &component.add {
                for part in &tensor.parts {
                    parts.insert(
                        (
                            component.target.clone(),
                            tensor.key.clone(),
                            part.role.clone(),
                        ),
                        part.clone(),
                    );
                }
            }
        }
        Self { parts }
    }

    /// The current writer's additions, for a caller that holds no `Begin`.
    pub fn load(meta: &Meta, transaction: &str, writer_session: u64) -> Result<Self> {
        let transaction = transaction_id(transaction)?;
        let mut rows: Vec<_> = meta.derived_row(&transaction)?.into_iter().collect();
        let row = open_row(&mut rows, &transaction, writer_session)?;
        Ok(Self::of(&declaration(row)?))
    }

    fn get(&self, (component, tensor_key, role): (&str, &str, &str)) -> Result<&PartDeclaration> {
        let part = self
            .parts
            .get(&(
                component.to_string(),
                tensor_key.to_string(),
                role.to_string(),
            ))
            .ok_or_else(|| Refusal {
                code: Code::MISSING_FIELD,
                detail: format!("{component}/{tensor_key}#{role} is not a declared addition"),
            })?;
        if part.source.is_some() {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "grafted part cannot receive byte writes",
            );
        }
        Ok(part)
    }
}

/// The exact native payload limit, checked against the current writer before
/// an embedding owner admits storage or asks a producer to supply any bytes.
pub fn part_write_bound(
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    additions: &Additions,
    value: (&str, &str, &str),
) -> Result<u64> {
    let transaction = transaction_id(transaction)?;
    meta.check_session(&transaction, writer_session)?;
    let part = additions.get(value)?;
    checked_bytes("derived part write", &part.shape, part.dtype)
}

pub fn config_write_bound(
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    name: &str,
) -> Result<u64> {
    let transaction = transaction_id(transaction)?;
    let mut rows: Vec<_> = meta.derived_row(&transaction)?.into_iter().collect();
    let row = open_row(&mut rows, &transaction, writer_session)?;
    let declaration = declaration(row)?;
    require_added_config(&declaration, name)?;
    Ok(declaration.max_new_bytes.min(limits::DOC_MAX_BYTES as u64))
}

fn require_added_config(declaration: &Declaration, name: &str) -> Result<()> {
    if declaration.configs.iter().any(|config| {
        matches!(config, ConfigDeclaration::Add { target }
            | ConfigDeclaration::Derive { target, .. } if target == name)
    }) {
        Ok(())
    } else {
        refuse(
            Code::MISSING_FIELD,
            format!("{name:?} is not a declared added config"),
        )
    }
}

/// Accept one part for a caller without the writer's `Additions`; see [`add_indexed_part`].
pub fn add_part<R: Read>(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    value: (&str, &str, &str),
    reader: &mut R,
) -> Result<crate::checkpoint::Written> {
    let additions = Additions::load(meta, transaction, writer_session)?;
    add_indexed_part(
        store,
        meta,
        transaction,
        writer_session,
        &additions,
        None,
        value,
        reader,
    )
}

/// Accept one part. The cost is this part's bytes plus one indexed SQL row: the
/// declaration is not re-read and earlier accepted parts are neither read nor rewritten.
#[allow(clippy::too_many_arguments)]
pub fn add_indexed_part<R: Read>(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    additions: &Additions,
    cursor: Option<&mut CheckpointCursor>,
    value: (&str, &str, &str),
    reader: &mut R,
) -> Result<crate::checkpoint::Written> {
    let (component, tensor_key, role) = value;
    let transaction = transaction_id(transaction)?;
    meta.check_session(&transaction, writer_session)?;
    let expected = additions.get(value)?;

    let written = checkpoint::objectize(
        store,
        &format!("{component}/{tensor_key}#{role}"),
        expected.dtype,
        expected.shape.clone(),
        reader,
    )?;
    let mut extra = [0u8; 1];
    if reader.read(&mut extra).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read addition tail: {error}"),
    })? != 0
    {
        return refuse(
            Code::LENGTH_MISMATCH,
            format!("{component}/{tensor_key}#{role} produced bytes beyond its declared shape"),
        );
    }
    for object in written.part.segments() {
        // An admitted object already has a fresh record. A deduplicated destination is
        // checked here, so a same-key corrupt resident file cannot enter the commit.
        let verified = store.open_verified(&object.sha256)?;
        if verified.len() != object.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!("{} verified length changed after admission", object.id()),
            );
        }
    }
    // The private root names the objects before SQL accepts the part, so GC never
    // observes an accepted part whose bytes it may reclaim.
    roots::append_parts(store, &transaction, written.part.segments())?;
    let added = AddedPart {
        component: component.to_string(),
        key: tensor_key.to_string(),
        role: role.to_string(),
        part: written.part.clone(),
    };
    if meta.accept_part(&transaction, writer_session, &added)? {
        if let Some(cursor) = cursor {
            cursor.pending.push(added);
        }
    }
    Ok(written)
}

pub fn add_config<R: Read>(
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
    config_name: &str,
    reader: &mut R,
) -> Result<()> {
    let transaction = transaction_id(transaction)?;
    let maximum = {
        let mut txn = meta.txn_for(&transaction)?;
        let row = open_row(&mut txn.rows, &transaction, writer_session)?;
        let declaration = declaration(row)?;
        require_added_config(&declaration, config_name)?;
        declaration.max_new_bytes.min(limits::DOC_MAX_BYTES as u64)
    };

    let mut input = Vec::new();
    reader
        .take(maximum + 1)
        .read_to_end(&mut input)
        .map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read config: {error}"),
        })?;
    if input.len() as u64 > maximum {
        return refuse(
            Code::SIZE_CAP,
            "config exceeds the output or document byte cap",
        );
    }
    let config = crate::header::canonical_config(&format!("config {config_name:?}"), &input)?;

    let mut txn = meta.txn_for(&transaction)?;
    let row = open_row(&mut txn.rows, &transaction, writer_session)?;
    if let Some(existing) = row
        .added_configs
        .iter()
        .find(|added| added.name == config_name)
    {
        if existing.config == config {
            return Ok(());
        }
        return refuse(
            Code::TRANSACTION_CONFLICT,
            format!("added config {config_name:?} differs from accepted bytes"),
        );
    }
    row.added_configs.push(AddedConfig {
        name: config_name.to_string(),
        config,
    });
    row.normalize();
    let declaration = declaration(row)?;
    let (planned, _) = declaration.planned_additions()?;
    let config_bytes: u64 = row
        .added_configs
        .iter()
        .map(|c| c.config.len() as u64)
        .sum();
    if !declaration.files.is_empty()
        && planned
            .checked_add(config_bytes)
            .is_none_or(|n| n > declaration.max_new_bytes)
    {
        return refuse(
            Code::QUOTA_EXHAUSTED,
            "configs and companion/tensor additions exceed output grant",
        );
    }
    txn.commit()?;
    Ok(())
}
#[derive(Debug, Clone)]
pub struct ReceiptFacts {
    pub transaction: String,
    pub declaration: Declaration,
    pub sources: Vec<SourceFact>,
    pub header: ObjectRef,
    pub manifest: ObjectRef,
    pub added_objects: Vec<ObjectRef>,
    pub inherited_payload_objects: u64,
    pub inherited_payload_bytes: u64,
}

impl ReceiptFacts {
    /// The receipt names its declaration by digest and length: the declaration grows with the
    /// tensors declared, and the receipt stays constant.
    pub fn to_value(&self) -> Value {
        let bytes = crate::canon::write(&self.declaration.to_value());
        let Value::Obj(mut fields) = self.facts() else {
            unreachable!("receipt facts are an object")
        };
        fields.push((
            "declaration_digest".into(),
            Value::str(ObjectRef::of(&bytes).id()),
        ));
        fields.push(("declaration_length".into(), Value::uint(bytes.len() as u64)));
        fields.sort_by(|left, right| left.0.cmp(&right.0));
        Value::Obj(fields)
    }

    fn facts(&self) -> Value {
        Value::obj(vec![
            (
                "added_objects",
                Value::arr(self.added_objects.iter().map(ObjectRef::to_value).collect()),
            ),
            ("header", self.header.to_value()),
            (
                "inherit_observation",
                Value::obj(vec![
                    ("bytes", Value::uint(self.inherited_payload_bytes)),
                    ("hashes", Value::uint(0)),
                    ("objects", Value::uint(self.inherited_payload_objects)),
                    ("reads", Value::uint(0)),
                ]),
            ),
            ("manifest", self.manifest.to_value()),
            (
                "sources",
                Value::arr(
                    self.sources
                        .iter()
                        .map(|source| {
                            Value::obj(vec![
                                ("alias", Value::str(source.alias.clone())),
                                (
                                    "components",
                                    Value::arr(
                                        source
                                            .components
                                            .iter()
                                            .map(|component| Value::str(component.clone()))
                                            .collect(),
                                    ),
                                ),
                                ("header", source.header.to_value()),
                                ("manifest", source.manifest.to_value()),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("transaction_id", Value::str(self.transaction.clone())),
        ])
    }
}

fn added_objects(row: &TransactionRow) -> Vec<ObjectRef> {
    let mut objects: Vec<ObjectRef> = row
        .added_parts
        .iter()
        .flat_map(|added| added.part.segments().iter().cloned())
        .collect();
    if matches!(row.state, TransactionState::Committed(_)) {
        if let Ok(declaration) = declaration(row) {
            objects.extend(
                declaration
                    .files
                    .iter()
                    .map(|(_, data)| ObjectRef::of(data)),
            );
        }
    }
    objects.sort_by(|a, b| (&a.sha256, a.length).cmp(&(&b.sha256, b.length)));
    objects.dedup();
    objects
}

fn inherit_observation(header: &Header, row: &TransactionRow) -> Result<(usize, u64)> {
    let added_set: HashSet<(String, u64)> = added_objects(row)
        .into_iter()
        .map(|object| (object.sha256, object.length))
        .collect();
    let mut inherited: Vec<ObjectRef> = header
        .tensors()
        .flat_map(|(_, _, tensor)| tensor.parts.iter())
        .flat_map(|(_, part)| part.segments().iter().cloned())
        .filter(|object| !added_set.contains(&(object.sha256.clone(), object.length)))
        .collect();
    inherited.sort_by(|a, b| (&a.sha256, a.length).cmp(&(&b.sha256, b.length)));
    inherited.dedup();
    let bytes = inherited.iter().try_fold(0u64, |total, object| {
        total
            .checked_add(object.length)
            .ok_or_else(|| arithmetic("inherited payload byte total overflows"))
    })?;
    Ok((inherited.len(), bytes))
}

fn receipt_from(_store: &Store, row: &TransactionRow) -> Result<ReceiptFacts> {
    let declaration = declaration(row)?;
    let header = row.header.clone().ok_or_else(|| Refusal {
        code: Code::ARTIFACT_INCOMPLETE,
        detail: format!("committed transaction {} has no header", row.id),
    })?;
    let manifest = row.manifest.clone().ok_or_else(|| Refusal {
        code: Code::ARTIFACT_INCOMPLETE,
        detail: format!("committed transaction {} has no manifest", row.id),
    })?;
    Ok(ReceiptFacts {
        transaction: row.id.clone(),
        declaration,
        sources: row.source_facts.clone(),
        header,
        manifest,
        added_objects: added_objects(row),
        inherited_payload_objects: row.inherited_payload_objects,
        inherited_payload_bytes: row.inherited_payload_bytes,
    })
}

/// The caller retains Begin's writer hold and source leases through this call.
/// Root membership was admitted at begin; the live hold now protects those exact
/// bytes even if their original repository/session/private root is released.
pub fn commit(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    writer_session: u64,
) -> Result<ReceiptFacts> {
    let transaction = transaction_id(transaction)?;
    let row_before = {
        let mut txn = meta.txn_for(&transaction)?;
        open_row(&mut txn.rows, &transaction, writer_session)?.clone()
    };
    let declaration = declaration(&row_before)?;
    let sources = load_sources(store, &declaration)?;
    let (header, closure, registry_adoptions) = build_header(
        &declaration,
        &sources,
        &row_before.added_parts,
        &row_before.added_configs,
        false,
    )?;
    admit_registry(store, &registry_adoptions)?;
    let header_ref = checkpoint::put_doc(store, &header)?;
    let mut manifest = Draft::of(&build_manifest(&header, &header_ref, &closure)?);
    for (_, data) in &declaration.files {
        let expected = ObjectRef::of(data);
        store.put_stream(
            &mut &data[..],
            Some(&expected),
            &crate::store::Fault::default(),
        )?;
    }
    manifest
        .entries
        .extend(companion_files(&declaration, &sources)?);
    manifest.entries.sort_by(|a, b| a.0.cmp(&b.0));
    let manifest = manifest.seal()?;
    let manifest_ref = store.put_manifest(&manifest)?.obj;
    store.read_manifest(&manifest_ref)?;
    let walk = checkpoint::walk(store, &manifest)?;
    walk.require_held(store, &crate::ingest::custody::live(store.root())?)?;
    let (inherited_payload_objects, inherited_payload_bytes) =
        inherit_observation(&header, &row_before)?;
    store.sync_admitted()?;

    let mut txn = meta.txn_for(&transaction)?;
    let row = open_row(&mut txn.rows, &transaction, writer_session)?;
    if row.added_parts != row_before.added_parts || row.added_configs != row_before.added_configs {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "the addition set changed while commit constructed the target; retry commit",
        );
    }
    roots::pending(store, &transaction, &manifest_ref)?;
    row.state = TransactionState::Committed(Disposition::Pending);
    row.current_session = None;
    row.header = Some(header_ref);
    row.manifest = Some(manifest_ref);
    row.inherited_payload_objects = inherited_payload_objects as u64;
    row.inherited_payload_bytes = inherited_payload_bytes;
    let committed = row.clone();
    txn.commit()?;
    receipt_from(store, &committed)
}

#[derive(Debug, Clone)]
pub enum Lookup {
    Absent,
    Open { writer_session: Option<u64> },
    Committed(Box<DispositionResult>),
    Abandoned,
}

pub fn lookup(store: &Store, meta: &Meta, transaction: &str) -> Result<Lookup> {
    let transaction = transaction_id(transaction)?;
    let Some(ref row) = meta.derived_row(&transaction)? else {
        return Ok(Lookup::Absent);
    };
    match row.state {
        TransactionState::Open => Ok(Lookup::Open {
            writer_session: row.current_session,
        }),
        TransactionState::Committed(ref recorded) => {
            let disposition = effective_disposition(store, row, recorded)?;
            Ok(Lookup::Committed(Box::new(DispositionResult {
                receipt: receipt_from(store, row)?,
                disposition,
            })))
        }
        TransactionState::Abandoned => Ok(Lookup::Abandoned),
    }
}

pub fn fence(meta: &Meta, transaction: &str, writer_session: u64) -> Result<bool> {
    let transaction = transaction_id(transaction)?;
    let mut txn = meta.txn_for(&transaction)?;
    let row = txn
        .rows
        .iter_mut()
        .find(|row| row.id == transaction)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no derived transaction {transaction}"),
        })?;
    if row.state != TransactionState::Open || row.current_session != Some(writer_session) {
        return Ok(false);
    }
    row.current_session = None;
    txn.commit()?;
    Ok(true)
}

pub fn abandon(store: &Store, meta: &Meta, transaction: &str) -> Result<Option<ReceiptFacts>> {
    let transaction = transaction_id(transaction)?;
    let mut txn = meta.txn_for(&transaction)?;
    let result = match txn.rows.iter_mut().find(|row| row.id == transaction) {
        None => {
            txn.rows.push(TransactionRow {
                id: transaction.clone(),
                declaration: None,
                state: TransactionState::Abandoned,
                highest_session: 0,
                current_session: None,
                added_parts: Vec::new(),
                added_configs: Vec::new(),
                source_facts: Vec::new(),
                inherited_payload_objects: 0,
                inherited_payload_bytes: 0,
                header: None,
                manifest: None,
            });
            None
        }
        Some(row) => match row.state {
            TransactionState::Abandoned => None,
            TransactionState::Committed(_) => Some(row.clone()),
            TransactionState::Open if row.current_session.is_some() => {
                return refuse(
                    Code::STORE_BUSY,
                    format!(
                        "{} still has active WriterSessionId {:?}; fence it before abandonment",
                        row.id, row.current_session
                    ),
                )
            }
            TransactionState::Open => {
                row.state = TransactionState::Abandoned;
                row.added_parts.clear();
                row.added_configs.clear();
                None
            }
        },
    };
    if let Some(row) = result {
        drop(txn);
        return Ok(Some(receipt_from(store, &row)?));
    }
    txn.commit()?;
    roots::remove(store, &transaction)?;
    Ok(None)
}

#[derive(Debug, Clone)]
pub struct DispositionResult {
    receipt: ReceiptFacts,
    disposition: Disposition,
}

impl DispositionResult {
    pub fn to_value(&self) -> Value {
        let disposition = self.disposition.to_value();
        Value::obj(vec![
            ("disposition", disposition),
            ("receipt", self.receipt.to_value()),
        ])
    }

    pub fn disposition_value(&self) -> Value {
        self.disposition.to_value()
    }

    pub fn receipt(&self) -> &ReceiptFacts {
        &self.receipt
    }
}

/// Retain a committed result under an opaque private root. The name is supplied by the
/// RecordOwner; TensorFS neither derives that name nor fabricates a model repository.
pub fn adopt(
    store: &Store,
    meta: &Meta,
    transaction: &str,
    scratch_root_id: &str,
) -> Result<DispositionResult> {
    let transaction = transaction_id(transaction)?;
    name("private scratch root", scratch_root_id)?;
    let mut txn = meta.txn()?;
    let index = txn
        .rows
        .iter()
        .position(|row| row.id == transaction)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no derived transaction {transaction}"),
        })?;
    let desired = Disposition::Adopted(scratch_root_id.into());
    match &txn.rows[index].state {
        TransactionState::Committed(Disposition::Pending) => {}
        TransactionState::Committed(existing) if existing == &desired => {
            let row = txn.rows[index].clone();
            roots::adopt(
                store,
                &transaction,
                row.manifest.as_ref().expect("committed manifest"),
                scratch_root_id,
            )?;
            drop(txn);
            return Ok(DispositionResult {
                receipt: receipt_from(store, &row)?,
                disposition: desired,
            });
        }
        TransactionState::Committed(_) => {
            return refuse(
                Code::DISPOSITION_CONFLICT,
                "derived result was adopted under another name or already released",
            )
        }
        _ => {
            return refuse(
                Code::TRANSACTION_CLOSED,
                "only a committed result can be adopted",
            )
        }
    }
    if txn.rows.iter().any(|row| {
        row.id != transaction && row.state == TransactionState::Committed(desired.clone())
    }) {
        return refuse(
            Code::DISPOSITION_CONFLICT,
            "private scratch root already names another transaction",
        );
    }
    roots::adopt(
        store,
        &transaction,
        txn.rows[index].manifest.as_ref().ok_or_else(|| Refusal {
            code: Code::ARTIFACT_INCOMPLETE,
            detail: "committed transaction has no manifest".into(),
        })?,
        scratch_root_id,
    )?;
    txn.rows[index].state = TransactionState::Committed(desired.clone());
    let row = txn.rows[index].clone();
    txn.commit()?;
    Ok(DispositionResult {
        receipt: receipt_from(store, &row)?,
        disposition: desired,
    })
}

fn effective_disposition(
    store: &Store,
    row: &TransactionRow,
    recorded: &Disposition,
) -> Result<Disposition> {
    if recorded == &Disposition::Released {
        return Ok(Disposition::Released);
    }
    match roots::read(store, &row.id)? {
        Some(root) => {
            if row.manifest != root.manifest {
                return refuse(
                    Code::DISPOSITION_CONFLICT,
                    "private root differs from its committed receipt",
                );
            }
            Ok(root.name.map_or(Disposition::Pending, Disposition::Adopted))
        }
        None if recorded == &Disposition::Pending => Ok(Disposition::Pending),
        None => refuse(Code::DURABILITY_UNPROVEN, "adopted private root is absent"),
    }
}

pub fn dispose(store: &Store, meta: &Meta, transaction: &str) -> Result<DispositionResult> {
    let transaction = transaction_id(transaction)?;
    let mut txn = meta.txn_for(&transaction)?;
    let index = txn
        .rows
        .iter()
        .position(|row| row.id == transaction)
        .ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: format!("no derived transaction {transaction}"),
        })?;
    txn.rows[index].manifest.as_ref().ok_or_else(|| Refusal {
        code: Code::TRANSACTION_CLOSED,
        detail: format!("{transaction} is not committed"),
    })?;
    let desired = Disposition::Released;
    match &txn.rows[index].state {
        TransactionState::Committed(Disposition::Pending | Disposition::Adopted(_)) => {}
        TransactionState::Committed(existing) if existing == &desired => {
            let row = txn.rows[index].clone();
            drop(txn);
            roots::remove(store, &transaction)?;
            return Ok(DispositionResult {
                receipt: receipt_from(store, &row)?,
                disposition: desired,
            });
        }
        TransactionState::Committed(existing) => {
            return refuse(
                Code::DISPOSITION_CONFLICT,
                format!("{transaction} already has disposition {existing:?}, cannot release"),
            )
        }
        _ => {
            return refuse(
                Code::TRANSACTION_CLOSED,
                format!("{transaction} is not committed"),
            )
        }
    }

    txn.rows[index].state = TransactionState::Committed(desired.clone());
    let row = txn.rows[index].clone();
    txn.commit()?;
    roots::remove(store, &transaction)?;
    Ok(DispositionResult {
        receipt: receipt_from(store, &row)?,
        disposition: desired,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{Closure, Part, Tensor};
    use crate::ids::Doc;

    pub(super) fn created_declaration() -> Declaration {
        let encoding = registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec
            .object_id();
        Declaration {
            files: Vec::new(),
            objects: Vec::new(),
            work_fingerprint: Some(format!("sha256:{}", "44".repeat(32))),
            sources: Vec::new(),
            components: vec![ComponentDeclaration {
                target: "model".into(),
                source: None,
                source_component: None,
                drop: Vec::new(),
                add: vec![TensorDeclaration {
                    key: "weight".into(),
                    dtype: Dtype::F32,
                    shape: vec![512],
                    encoding,
                    parts: vec![PartDeclaration {
                        role: "value".into(),
                        dtype: Dtype::F32,
                        shape: vec![512],
                        source: None,
                    }],
                }],
            }],
            configs: Vec::new(),
            order: vec![("model".into(), "weight".into())],
            max_new_bytes: 2048,
        }
    }

    fn inherited_declaration(manifest: ObjectRef) -> Declaration {
        Declaration {
            files: Vec::new(),
            objects: Vec::new(),
            work_fingerprint: Some(format!("sha256:{}", "44".repeat(32))),
            sources: vec![Source {
                alias: "source".into(),
                manifest,
            }],
            components: vec![ComponentDeclaration {
                target: "inherited".into(),
                source: Some("source".into()),
                source_component: Some("model".into()),
                drop: Vec::new(),
                add: Vec::new(),
            }],
            configs: Vec::new(),
            order: vec![("inherited".into(), "weight".into())],
            max_new_bytes: 0,
        }
    }

    #[test]
    fn failed_commit_retains_input_custody_until_retry_and_adoption() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-derived-commit-window-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let source_id = format!("sha256:{}", "91".repeat(32));
        let source_writer =
            begin(&store, &meta, &source_id, 1, created_declaration(), None).unwrap();
        let source_bytes = vec![0x21; 2048];
        let source_part = add_part(
            &store,
            &meta,
            &source_id,
            1,
            ("model", "weight", "value"),
            &mut source_bytes.as_slice(),
        )
        .unwrap();
        let source = commit(&store, &meta, &source_id, 1).unwrap();
        release_guards(
            &meta,
            Some(source_writer.writer_hold),
            source_writer.source_leases,
        );

        let id = format!("sha256:{}", "92".repeat(32));
        let mut declaration = inherited_declaration(source.manifest.clone());
        declaration.components[0].drop = vec!["weight".into()];
        declaration.components[0].add = created_declaration().components.remove(0).add;
        declaration.max_new_bytes = 2048;
        let writer = begin(&store, &meta, &id, 1, declaration.clone(), None).unwrap();
        dispose(&store, &meta, &source_id).unwrap();
        let target_bytes = vec![0x31; 2048];
        add_part(
            &store,
            &meta,
            &id,
            1,
            ("inherited", "weight", "value"),
            &mut target_bytes.as_slice(),
        )
        .unwrap();
        let before = roots::read(&store, &id).unwrap().unwrap();

        let connection = rusqlite::Connection::open(crate::catalog::Catalog::path(&root)).unwrap();
        connection.execute_batch("CREATE TRIGGER fail_derived_commit BEFORE DELETE ON tensorfs_derived_transactions BEGIN SELECT RAISE(ABORT,'derived commit window'); END").unwrap();
        let error = commit(&store, &meta, &id, 1).unwrap_err();
        assert_eq!(error.code, Code::IO_FAILED);
        assert!(error.detail.contains("derived commit window"));
        connection
            .execute_batch("DROP TRIGGER fail_derived_commit")
            .unwrap();
        drop(connection);
        assert!(matches!(
            lookup(&store, &meta, &id).unwrap(),
            Lookup::Open { .. }
        ));
        let pending = roots::read(&store, &id).unwrap().unwrap();
        assert_eq!(pending.sources, before.sources);
        assert_eq!(pending.blobs, before.blobs);
        assert!(pending.manifest.is_some());

        fence(&meta, &id, 1).unwrap();
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        crate::gc::collect(&root, false).unwrap();
        store.read_manifest(&source.manifest).unwrap();
        assert!(store.contains(&source_part.part.segments()[0].sha256));
        let replacement = begin(&store, &meta, &id, 2, declaration, None).unwrap();
        assert_eq!(completed(&meta, &id, 2).unwrap().0.len(), 1);
        let receipt = commit(&store, &meta, &id, 2).unwrap();
        assert_eq!(Some(receipt.manifest.clone()), pending.manifest);
        release_guards(
            &meta,
            Some(replacement.writer_hold),
            replacement.source_leases,
        );
        adopt(&store, &meta, &id, "commit-window-output").unwrap();
        crate::gc::collect(&root, false).unwrap();
        store.read_manifest(&receipt.manifest).unwrap();
        assert!(!store.contains(&source_part.part.segments()[0].sha256));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn retained_source_admission_and_writer_hold_share_the_gc_roots() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-retained-source-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let source_id = format!("sha256:{}", "81".repeat(32));
        let source_writer =
            begin(&store, &meta, &source_id, 1, created_declaration(), None).unwrap();
        let bytes = vec![0x21; 2048];
        add_part(
            &store,
            &meta,
            &source_id,
            1,
            ("model", "weight", "value"),
            &mut bytes.as_slice(),
        )
        .unwrap();
        let source = commit(&store, &meta, &source_id, 1).unwrap();
        release_guards(
            &meta,
            Some(source_writer.writer_hold),
            source_writer.source_leases,
        );
        let preview = || {
            inspect_source(
                &store,
                &meta,
                source.manifest.clone(),
                vec!["model".into()],
                Vec::new(),
            )
        };

        // Committed pending output and owner-adopted output have real filesystem roots.
        preview().unwrap();
        adopt(&store, &meta, &source_id, "source-private").unwrap();
        preview().unwrap();
        dispose(&store, &meta, &source_id).unwrap();
        assert_eq!(preview().unwrap_err().code, Code::ROOT_ABSENT);
        let target_id = format!("sha256:{}", "82".repeat(32));
        assert_eq!(
            begin(
                &store,
                &meta,
                &target_id,
                1,
                inherited_declaration(source.manifest.clone()),
                None,
            )
            .err()
            .unwrap()
            .code,
            Code::ROOT_ABSENT
        );
        assert!(matches!(
            lookup(&store, &meta, &target_id).unwrap(),
            Lookup::Absent
        ));

        // The coordinator's actual prepared-source lifecycle names a private candidate.
        let operation = "prepared-source";
        let request_digest = format!("sha256:{}", "83".repeat(32));
        let catalog = crate::catalog::Catalog::open(&root).unwrap();
        let crate::catalog::SourcePreparation::Open(source_guard) = catalog
            .begin_source_preparation(operation, &request_digest)
            .unwrap()
        else {
            panic!("source operation already prepared")
        };
        crate::ingest::transaction::ensure_session_root(&root, operation, "_tensorfs").unwrap();
        crate::ingest::transaction::name_candidates(
            &root,
            operation,
            std::slice::from_ref(&source.manifest),
        )
        .unwrap();
        // An actual candidate is retained even before the rebuildable result row is written.
        preview().unwrap();
        catalog
            .commit_source_preparation(
                operation,
                &request_digest,
                &[crate::catalog::SourcePreparationResult {
                    slot: "source".into(),
                    profile: "proof".into(),
                    manifest_sha256: source.manifest.sha256.clone(),
                    manifest_length: source.manifest.length,
                }],
            )
            .unwrap();
        drop(source_guard);
        crate::gc::collect(&root, false).unwrap();
        preview().unwrap();
        let mut wrong = inherited_declaration(source.manifest.clone());
        wrong.sources[0].manifest.length += 1;
        assert_eq!(
            begin(&store, &meta, &target_id, 1, wrong, None)
                .err()
                .unwrap()
                .code,
            Code::ROOT_ABSENT
        );

        let writer = begin(
            &store,
            &meta,
            &target_id,
            1,
            inherited_declaration(source.manifest.clone()),
            None,
        )
        .unwrap();
        crate::ingest::source::release_model_source(&store, operation).unwrap();
        // The resumable writer now retains its admitted sources independently of
        // their original source operation, including after the live writer exits.
        preview().unwrap();
        writer.writer_hold.still_registered(&meta).unwrap();
        assert_eq!(
            crate::gc::collect(&root, false).unwrap_err().code,
            Code::STORE_BUSY
        );
        let inherited = commit(&store, &meta, &target_id, 1).unwrap();
        assert_eq!(inherited.inherited_payload_bytes, 2048);
        assert!(inherited.added_objects.is_empty());
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        crate::gc::collect(&root, false).unwrap();
        store.read_manifest(&inherited.manifest).unwrap();
        inspect_source(
            &store,
            &meta,
            inherited.manifest.clone(),
            vec!["inherited".into()],
            Vec::new(),
        )
        .unwrap();
        dispose(&store, &meta, &target_id).unwrap();
        assert_eq!(
            inspect_source(
                &store,
                &meta,
                inherited.manifest,
                vec!["inherited".into()],
                Vec::new()
            )
            .unwrap_err()
            .code,
            Code::ROOT_ABSENT
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn retained_source_still_refuses_corrupt_payload_and_abandoned_writer() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-retained-source-refusal-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let source_id = format!("sha256:{}", "84".repeat(32));
        let source_writer =
            begin(&store, &meta, &source_id, 1, created_declaration(), None).unwrap();
        let bytes = vec![0x31; 2048];
        let part = add_part(
            &store,
            &meta,
            &source_id,
            1,
            ("model", "weight", "value"),
            &mut bytes.as_slice(),
        )
        .unwrap();
        let source = commit(&store, &meta, &source_id, 1).unwrap();
        release_guards(
            &meta,
            Some(source_writer.writer_hold),
            source_writer.source_leases,
        );
        let target_id = format!("sha256:{}", "85".repeat(32));
        let writer = begin(
            &store,
            &meta,
            &target_id,
            1,
            inherited_declaration(source.manifest.clone()),
            None,
        )
        .unwrap();
        assert!(fence(&meta, &target_id, 1).unwrap());
        abandon(&store, &meta, &target_id).unwrap();
        assert_eq!(
            commit(&store, &meta, &target_id, 1).unwrap_err().code,
            Code::TRANSACTION_CLOSED
        );
        release_guards(&meta, Some(writer.writer_hold), writer.source_leases);
        let path = store.blob_path(&part.part.segments()[0].sha256);
        std::fs::remove_file(&path).unwrap();
        std::fs::write(path, vec![0x32; 2048]).unwrap();
        assert_eq!(
            inspect_source(
                &store,
                &meta,
                source.manifest,
                vec!["model".into()],
                Vec::new()
            )
            .unwrap_err()
            .code,
            Code::OBJECT_CORRUPT
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn private_adoption_retains_real_payload_until_dispose_without_a_repository() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-private-adopt-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let id = format!("sha256:{}", "55".repeat(32));
        let writer = begin(&store, &meta, &id, 1, created_declaration(), None).unwrap();
        let bytes = vec![0x11; 2048];
        let written = add_part(
            &store,
            &meta,
            &id,
            1,
            ("model", "weight", "value"),
            &mut bytes.as_slice(),
        )
        .unwrap();
        let receipt = commit(&store, &meta, &id, 1).unwrap();
        writer.writer_hold.release(&meta).unwrap();
        let payload = &written.part.segments()[0];

        // The owner has not answered yet: even this pending result is rooted after its
        // writer is gone, closing the commit-to-finalize GC window.
        crate::gc::collect(&root, false).unwrap();
        assert!(store.contains(&payload.sha256));
        store.read_manifest(&receipt.manifest).unwrap();
        // Crash after the private name reached disk but before its SQL lifecycle update.
        roots::adopt(&store, &id, &receipt.manifest, "private-root").unwrap();
        let Lookup::Committed(observed) = lookup(&store, &meta, &id).unwrap() else {
            panic!("committed output missing")
        };
        assert_eq!(
            observed.disposition_value(),
            Disposition::Adopted("private-root".into()).to_value()
        );
        assert_eq!(
            adopt(&store, &meta, &id, "different-root")
                .unwrap_err()
                .code,
            Code::DISPOSITION_CONFLICT
        );
        let adopted = adopt(&store, &meta, &id, "private-root").unwrap();
        assert_eq!(adopted.receipt().to_value(), receipt.to_value());
        let reopened = Meta::open(&store).unwrap();
        assert_eq!(
            adopt(&store, &reopened, &id, "private-root")
                .unwrap()
                .to_value(),
            adopted.to_value()
        );
        assert_eq!(
            adopt(&store, &reopened, &id, "different-root")
                .unwrap_err()
                .code,
            Code::DISPOSITION_CONFLICT
        );
        crate::gc::collect(&root, false).unwrap();
        assert!(store.contains(&payload.sha256));
        store.read_manifest(&receipt.manifest).unwrap();
        // Liveness is discoverable even when the transaction journal cannot be decoded.
        // A GC hook that consults SQLite rather than private root files fails this arm.
        let database = crate::catalog::Catalog::path(&root);
        let connection = rusqlite::Connection::open(&database).unwrap();
        let saved: Vec<u8> = connection
            .query_row(
                "SELECT bytes FROM tensorfs_derived_transactions WHERE id=?1",
                [&id],
                |row| row.get(0),
            )
            .unwrap();
        connection
            .execute(
                "UPDATE tensorfs_derived_transactions SET bytes=?1 WHERE id=?2",
                rusqlite::params![b"not a transaction".as_slice(), &id],
            )
            .unwrap();
        drop(connection);
        crate::gc::collect(&root, false).unwrap();
        assert!(store.contains(&payload.sha256));
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection
            .execute(
                "UPDATE tensorfs_derived_transactions SET bytes=?1 WHERE id=?2",
                rusqlite::params![saved, &id],
            )
            .unwrap();
        drop(connection);
        assert!(std::fs::read_dir(root.join("repos"))
            .map(|rows| rows.count() == 0)
            .unwrap_or(true));

        let saved_root = roots::read(&store, &id).unwrap().unwrap();
        let released = dispose(&store, &reopened, &id).unwrap();
        assert_eq!(released.receipt().to_value(), receipt.to_value());
        // Crash after the terminal journal update but before the root was removed.
        roots::write(&store, &saved_root).unwrap();
        let Lookup::Committed(observed) = lookup(&store, &reopened, &id).unwrap() else {
            panic!("committed receipt missing")
        };
        assert_eq!(
            observed.disposition_value(),
            Disposition::Released.to_value()
        );
        assert_eq!(
            dispose(&store, &reopened, &id).unwrap().to_value(),
            released.to_value()
        );
        assert!(roots::read(&store, &id).unwrap().is_none());
        assert_eq!(
            adopt(&store, &reopened, &id, "private-root")
                .unwrap_err()
                .code,
            Code::DISPOSITION_CONFLICT
        );
        crate::gc::collect(&root, false).unwrap();
        assert!(!store.contains(&payload.sha256));
        assert!(!store.manifest_path(&receipt.manifest.sha256).exists());
        assert_eq!(
            dispose(&store, &reopened, &id).unwrap().to_value(),
            released.to_value()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn absent_abandonment_is_a_durable_fence_against_a_late_begin() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-absent-abandon-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let id = format!("sha256:{}", "66".repeat(32));
        assert!(abandon(&store, &meta, &id).unwrap().is_none());
        let reopened = Meta::open(&store).unwrap();
        assert!(matches!(
            lookup(&store, &reopened, &id).unwrap(),
            Lookup::Abandoned
        ));
        assert_eq!(
            begin(&store, &reopened, &id, 1, created_declaration(), None)
                .err()
                .unwrap()
                .code,
            Code::TRANSACTION_CLOSED
        );
        assert_eq!(
            adopt(&store, &reopened, &id, "private-root")
                .unwrap_err()
                .code,
            Code::TRANSACTION_CLOSED
        );
        let _ = std::fs::remove_dir_all(root);
    }
    use crate::manifest::{Draft, Entry};
    use crate::repository::{Mutation, ReleaseLane, RepositoryName};
    use crate::store::Fault;
    use std::fs;

    #[test]
    fn derived_source_ignores_missing_snapshot_siblings() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-derived-runtime-closure-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let spec = crate::registry::seeds()
            .into_iter()
            .find(|seed| seed.alias == "plain/1")
            .unwrap()
            .spec;
        let header = Header {
            configs: Vec::new(),
            assets: Vec::new(),
            encodings: vec![spec.clone()],
            components: vec![(
                "model".into(),
                vec![(
                    "weight".into(),
                    Tensor {
                        dtype: Dtype::F32,
                        shape: vec![1],
                        encoding: spec.object_id(),
                        parts: vec![("value".into(), Part::plan(Dtype::F32, vec![1], &[0; 4]))],
                    },
                )],
            )],
        };
        header.validate(&Closure::default()).unwrap();
        let header_bytes = header.canonical_bytes().unwrap();
        let header_ref = store
            .put_stream(
                &mut header_bytes.as_slice(),
                Some(&ObjectRef::of(&header_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let readme_bytes = b"checkpoint notes";
        let readme = store
            .put_stream(
                &mut readme_bytes.as_slice(),
                Some(&ObjectRef::of(readme_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let sample_bytes = b"sample image";
        let sample = store
            .put_stream(
                &mut sample_bytes.as_slice(),
                Some(&ObjectRef::of(sample_bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let manifest = Draft {
            entries: vec![
                ("README.txt".into(), Entry::File(readme.clone())),
                (
                    "model.cozytensors".into(),
                    Entry::CozyTensors(header_ref.clone()),
                ),
                ("samples/example.png".into(), Entry::File(sample.clone())),
            ],
        }
        .seal()
        .unwrap();
        let manifest_ref = store.put_manifest(&manifest).unwrap().obj;
        let repo = RepositoryName::new("org", "derived-source").unwrap();
        let checkpointed = store
            .apply_repository(
                None,
                &Mutation::PutCheckpoint {
                    repo: repo.clone(),
                    manifest: manifest_ref.clone(),
                },
                &Fault::default(),
            )
            .unwrap()
            .unwrap();
        // An owner-retained checkpoint is a GC root before any public release exists.
        inspect_source(
            &store,
            &meta,
            manifest_ref.clone(),
            vec!["model".into()],
            Vec::new(),
        )
        .unwrap();
        store
            .apply_repository(
                Some(&checkpointed.canonical_bytes()),
                &Mutation::UpdateRelease {
                    expected_revision: 0,
                    repo,
                    remove: Vec::new(),
                    set: vec![ReleaseLane {
                        extra: Default::default(),
                        lane: "main".into(),
                        manifest: manifest_ref.clone(),
                    }],
                    version: "1.0.0".into(),
                },
                &Fault::default(),
            )
            .unwrap();

        fs::remove_file(store.blob_path(&readme.sha256)).unwrap();
        fs::remove_file(store.blob_path(&sample.sha256)).unwrap();
        assert_eq!(
            checkpoint::walk(&store, &manifest)
                .unwrap()
                .require_resident(&store)
                .unwrap_err()
                .code,
            Code::OBJECT_ABSENT
        );

        let transaction = format!("sha256:{}", "11".repeat(32));
        let begun = begin(
            &store,
            &meta,
            &transaction,
            1,
            Declaration {
                files: Vec::new(),
                objects: Vec::new(),
                work_fingerprint: Some(format!("sha256:{}", "44".repeat(32))),
                sources: vec![Source {
                    alias: "source".into(),
                    manifest: manifest_ref,
                }],
                components: vec![ComponentDeclaration {
                    target: "model".into(),
                    source: Some("source".into()),
                    source_component: Some("model".into()),
                    drop: Vec::new(),
                    add: Vec::new(),
                }],
                configs: Vec::new(),
                order: vec![("model".into(), "weight".into())],
                max_new_bytes: 0,
            },
            None,
        )
        .unwrap();
        assert_eq!(begun.source_views.len(), 1);
        assert_eq!(begun.source_leases.len(), 1);
        assert_eq!(begun.source_leases[0].1.objects(), &[header_ref]);
        for (_, lease) in begun.source_leases {
            lease.release(&meta).unwrap();
        }
        begun.writer_hold.release(&meta).unwrap();
        let _ = fs::remove_dir_all(root);
    }
}

#[cfg(test)]
mod resume_tests;

#[cfg(test)]
mod graft_tests;

#[cfg(test)]
mod companion_tests;

#[cfg(test)]
mod part_rows_tests;

#[cfg(test)]
mod group_commit_tests;

#[cfg(test)]
mod history_cost_tests;
