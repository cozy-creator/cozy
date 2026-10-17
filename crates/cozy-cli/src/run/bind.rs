//! Binds `cozy run`'s terms to a callable's interface: the payload, the files it attaches, the
//! model choices and a conversion's destination.
//!
//! Typing is the interface's, never how a token looks: `steps=20` is an int because `steps` is
//! declared int. Types, required fields and bounds are strict, and every problem is named in one
//! refusal. An undeclared field, at any depth, is dropped with one warning and never refuses a run.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read as _,
    path::{Path, PathBuf},
};

use cozy_proto::{
    error::{Error, Kind},
    ids::Digest,
};
use serde_json::{Map, Number, Value};
use sha2::{Digest as _, Sha256};

use super::{
    describe,
    interface::{AssetsSlot, Callable, Constraints, Field, Struct, Type, Wire},
};

/// A file with no declared bound is still bounded.
const DEFAULT_MAX_BYTES: u64 = 64 << 20;

/// The request field every callable takes beside its interface: how LoRAs apply.
const LORA_MODE: &str = "lora_mode";

/// What a run was given on the command line.
#[derive(Debug, Default)]
pub struct Terms {
    /// The terms after the target: `key=value`, `key:=json`, `model.<slot>=<ref>`, positionals.
    pub inputs: Vec<String>,
    /// `--input FILE`: the whole payload as JSON or YAML; inline terms override it.
    pub input_file: Option<PathBuf>,
    /// `--asset [field.path=|label=]FILE`.
    pub assets: Vec<String>,
    /// `--lora slot:component=ref[,strength]`.
    pub loras: Vec<String>,
    /// The run's idempotency key, which an omitted integer `seed` derives from.
    pub key: String,
    /// Where relative paths start, and what `~` means.
    pub cwd: PathBuf,
    pub user_home: PathBuf,
}

#[derive(Debug, Default, PartialEq)]
pub struct Bound {
    pub payload: Map<String, Value>,
    pub files: Vec<InputFile>,
    pub models: Vec<Choice>,
    /// A conversion's `org/model`, where the machine publishes the weights it writes.
    pub destination: Option<String>,
    /// Undeclared fields that were dropped, as people read their paths.
    pub ignored: Vec<String>,
}

/// A local file bound to a payload field, whose value is its digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFile {
    pub field: String,
    pub path: PathBuf,
    pub digest: Digest,
    pub length: u64,
    pub media_type: String,
    pub order: u32,
}

/// One model slot's choice: its reference as written, and the adapters applied to it in order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Choice {
    /// The root slot's parameter, or another callable's full slot path.
    pub slot: String,
    pub reference: Option<String>,
    pub adapters: Vec<Adapter>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adapter {
    pub component: Option<String>,
    pub reference: String,
    /// A canonical decimal.
    pub scale: String,
}

impl Bound {
    /// The warning for the fields that were dropped, if any.
    pub fn warning(&self, target: &str) -> Option<String> {
        (!self.ignored.is_empty()).then(|| {
            let noun = if self.ignored.len() == 1 {
                "field"
            } else {
                "fields"
            };
            format!(
                "ignored unknown {noun} {} — not in {target}'s interface",
                self.ignored.join(", ")
            )
        })
    }
}

impl Bound {
    /// The payload's canonical JSON: sorted keys, no whitespace, exact integers.
    pub fn canonical_payload(&self) -> Vec<u8> {
        canonical(&Value::Object(self.payload.clone())).into_bytes()
    }
}

/// JSON with sorted keys and no whitespace, whatever serde_json's map order.
fn canonical(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            let members: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}:{}", Value::String(k.clone()), canonical(&object[k])))
                .collect();
            format!("{{{}}}", members.join(","))
        }
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        other => other.to_string(),
    }
}

/// A warm run's model choices: `model.<slot>=`, `--lora` and a conversion's input model. It
/// carries no payload, so any other term is refused.
pub fn warm_choices(callable: &Callable, terms: &Terms) -> Result<Vec<Choice>, Error> {
    let mut binder = Binder {
        target: "",
        callable,
        terms,
        document: Map::new(),
        choices: BTreeMap::new(),
        files: Vec::new(),
    };
    let mut inputs = terms.inputs.clone();
    binder.conversion(&mut inputs)?;
    let positional = binder.read_terms(&inputs)?;
    binder.loras()?;
    if !positional.is_empty()
        || !binder.document.is_empty()
        || !binder.files.is_empty()
        || !terms.assets.is_empty()
        || terms.input_file.is_some()
    {
        return Err(usage(
            "warm_inputs",
            "--warm keeps a function ready and takes no inputs, only model.<slot>= and --lora choices",
        ));
    }
    Ok(binder.choices.into_values().collect())
}

/// Binds `terms` for `callable`; `target` names it in messages (`org/pkg/fn`).
pub fn bind(target: &str, callable: &Callable, terms: &Terms) -> Result<Bound, Error> {
    let mut binder = Binder {
        target,
        callable,
        terms,
        document: Map::new(),
        choices: BTreeMap::new(),
        files: Vec::new(),
    };
    let mut inputs = terms.inputs.clone();
    let destination = binder.conversion(&mut inputs)?;
    if let Some(path) = &terms.input_file {
        binder.load(path)?;
    }
    let positional = binder.read_terms(&inputs)?;
    binder.primary(positional)?;
    binder.loras()?;
    let files = binder.attach()?;
    binder.seed();
    let ignored = validate(target, callable, &mut binder.document)?;
    Ok(Bound {
        payload: binder.document,
        files,
        models: binder.choices.into_values().collect(),
        destination,
        ignored,
    })
}

struct Binder<'a> {
    target: &'a str,
    callable: &'a Callable,
    terms: &'a Terms,
    document: Map<String, Value>,
    choices: BTreeMap<String, Choice>,
    /// Positional local files, bound to media inputs by kind.
    files: Vec<PathBuf>,
}

impl Binder<'_> {
    /// A conversion reads `<input-model> [<org/model>]`: the first is its first slot's choice.
    fn conversion(&mut self, inputs: &mut Vec<String>) -> Result<Option<String>, Error> {
        if !self.callable.is_conversion() {
            return Ok(None);
        }
        let (bare, rest): (Vec<String>, Vec<String>) =
            inputs.drain(..).partition(|term| !term.contains('='));
        *inputs = rest;
        if bare.len() > 2 {
            return Err(usage(
                "conversion_positionals",
                format!(
                    "{} takes <{}> and an optional <org/model> destination; got {} positional values",
                    self.callable.name,
                    self.callable.models[0].param,
                    bare.len()
                ),
            )
            .remedy(describe::usage_line(self.target, self.callable)));
        }
        let mut bare = bare.into_iter();
        if let Some(input) = bare.next() {
            let slot = self.callable.models[0].param.clone();
            self.choose(&slot, input)?;
        }
        bare.next()
            .map(|destination| {
                let mut parts = destination.split('/');
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(org), Some(name), None) if !org.is_empty() && !name.is_empty() => {
                        Ok(destination.to_lowercase())
                    }
                    _ => Err(usage(
                        "destination_malformed",
                        format!("{destination:?} is not an org/model destination"),
                    )),
                }
            })
            .transpose()
    }

    /// `--input FILE`: JSON, or YAML by its name or when it is not JSON.
    fn load(&mut self, path: &Path) -> Result<(), Error> {
        let path = self.resolve(path);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            Error::new(
                Kind::NotFound,
                "input_unreadable",
                format!("--input {}: {e}", path.display()),
            )
            .remedy("--input takes one JSON or YAML file holding the whole payload")
        })?;
        let yaml = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"));
        let loaded: Value = match serde_json::from_str(&text) {
            Ok(value) if !yaml => value,
            _ => serde_yaml::from_str(&text).map_err(|e| {
                invalid("input_invalid", format!("--input {}: {e}", path.display()))
            })?,
        };
        let Value::Object(loaded) = loaded else {
            return Err(invalid(
                "input_invalid",
                format!("--input {} does not hold one object", path.display()),
            ));
        };
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        for (key, mut value) in loaded {
            if key == "models" {
                self.envelope(value)?;
                continue;
            }
            let key = self.fold(&key)?;
            if let Some(field) = self.callable.field(&key) {
                // A file named inside the file is relative to the file.
                relative_files(&field.ty, &mut value, &base);
            }
            self.document.insert(key, value);
        }
        Ok(())
    }

    /// `{"models": {"<slot>": {"ref": "…", "lora": [{"ref": "…", "weight": 0.8}]}}}`.
    fn envelope(&mut self, value: Value) -> Result<(), Error> {
        let malformed = || {
            usage("model_binding_invalid", "models must be an object keyed by model slot").remedy(
                r#"{"models": {"<slot>": {"ref": "org/model@release", "lora": [{"ref": "org/adapter", "weight": 0.8}]}}}"#,
            )
        };
        let Value::Object(slots) = value else {
            return Err(malformed());
        };
        for (slot, entry) in slots {
            if let Some(reference) = entry.get("ref").and_then(Value::as_str) {
                self.choose(&slot, reference.to_owned())?;
            }
            match entry.get("lora") {
                None => {}
                Some(Value::Array(list)) => self.overlays(&slot, list)?,
                Some(_) => return Err(malformed()),
            }
        }
        Ok(())
    }

    /// The argv terms; answers the bare positionals.
    fn read_terms(&mut self, inputs: &[String]) -> Result<Vec<String>, Error> {
        let mut positional = Vec::new();
        for term in inputs {
            if self.positional_file(term)? {
                self.files.push(self.resolve(Path::new(term)));
                continue;
            }
            let colon = term.find(":=");
            let equals = term.find('=');
            let (key, raw, structured) = match (colon, equals) {
                (Some(c), Some(e)) if e == c + 1 => (&term[..c], &term[c + 2..], true),
                (_, Some(e)) => (&term[..e], &term[e + 1..], false),
                _ => {
                    positional.push(term.clone());
                    continue;
                }
            };
            if let Some(slot) = key.strip_prefix("model.") {
                if structured {
                    return Err(usage(
                        "model_choice_malformed",
                        format!("{key} has no structured spelling"),
                    )
                    .remedy(format!("{key}=org/model[@release][/lane]")));
                }
                self.choose(slot, raw.trim().to_owned())?;
                continue;
            }
            let key = self.fold(key)?;
            let value = if structured {
                serde_json::from_str(raw).map_err(|_| {
                    usage("json_malformed", format!("{key}:=… is not JSON: {raw}")).remedy(
                        "`key:=<json>` carries a nested value verbatim; `key=value` is the scalar form",
                    )
                })?
            } else {
                match self.callable.field(&key) {
                    Some(field) => self.typed(field, raw)?,
                    None => Value::String(raw.to_owned()),
                }
            };
            self.document.insert(key, value);
        }
        Ok(positional)
    }

    /// One positional value fills the first declared field a text value can fill.
    fn primary(&mut self, positional: Vec<String>) -> Result<(), Error> {
        let Some(first) = positional.first() else {
            return Ok(());
        };
        let Some(field) = self
            .callable
            .request
            .fields
            .iter()
            .find(|f| !self.callable.is_assets(&f.name) && f.ty.media().is_none())
        else {
            return Err(usage(
                "positional_unbound",
                format!(
                    "{} has no request field a positional value fills",
                    self.callable.name
                ),
            ));
        };
        if positional.len() > 1 {
            return Err(usage(
                "positional_extra",
                format!(
                    "{} takes one positional value ({}); got {}",
                    self.callable.name,
                    field.name,
                    positional.len()
                ),
            )
            .remedy("every other field is `key=value` or `key:=<json>`"));
        }
        let value = self.typed(field, first)?;
        self.document.insert(field.name.clone(), value);
        Ok(())
    }

    /// `--lora slot[:component]=ref[,strength]`, in order; without a component, Runtime matches
    /// the adapter's own.
    fn loras(&mut self) -> Result<(), Error> {
        for raw in &self.terms.loras {
            let malformed = || {
                usage(
                    "lora_malformed",
                    "--lora expects slot[:component]=reference[,strength]",
                )
            };
            let (target, reference) = raw.split_once('=').ok_or_else(malformed)?;
            let (slot, component) = match target.split_once(':') {
                Some((_, component))
                    if component.is_empty() || component.contains([',', ' ', '\t']) =>
                {
                    return Err(malformed());
                }
                Some((slot, component)) => (slot, Some(component.to_owned())),
                None => (target, None),
            };
            let (reference, scale) = reference.split_once(',').unwrap_or((reference, "1"));
            let adapter = Adapter {
                component,
                reference: nonempty(reference, slot)?,
                scale: canonical_scale(scale)?,
            };
            self.choice(slot)?.adapters.push(adapter);
        }
        Ok(())
    }

    fn overlays(&mut self, slot: &str, list: &[Value]) -> Result<(), Error> {
        for entry in list {
            let reference = nonempty(entry["ref"].as_str().unwrap_or_default(), slot)?;
            let scale = match &entry["weight"] {
                Value::Number(n) => n.to_string(),
                Value::String(s) => s.clone(),
                _ => {
                    return Err(usage(
                        "model_overlay_invalid",
                        format!("model overlay for {slot} requires a numeric weight"),
                    ));
                }
            };
            let adapter = Adapter {
                component: entry["component"].as_str().map(str::to_owned),
                reference,
                scale: canonical_scale(&scale)?,
            };
            self.choice(slot)?.adapters.push(adapter);
        }
        Ok(())
    }

    fn choose(&mut self, slot: &str, reference: String) -> Result<(), Error> {
        let choice = self.choice(slot)?;
        if choice.reference.is_some() {
            return Err(usage(
                "model_slot_repeated",
                format!("model slot {slot} was chosen more than once"),
            ));
        }
        choice.reference = Some(reference);
        Ok(())
    }

    /// The choice for `asked`: a declared slot's parameter or path, or another callable's full
    /// slot path (`[org/pkg/]callable.models.param`), whose machine checks it.
    fn choice(&mut self, asked: &str) -> Result<&mut Choice, Error> {
        let slot = match self.callable.slot(asked) {
            Some(slot) => slot.param.clone(),
            None if child_slot(asked)
                && !asked.starts_with(&format!("{}.", self.callable.name)) =>
            {
                asked.to_owned()
            }
            None => {
                let declared: Vec<&str> = self
                    .callable
                    .models
                    .iter()
                    .map(|s| s.param.as_str())
                    .collect();
                let remedy = if declared.is_empty() {
                    format!("{} declares no model slots", self.callable.name)
                } else {
                    format!("{} declares: {}", self.callable.name, declared.join(", "))
                };
                return Err(usage(
                    "model_slot_unknown",
                    format!("no model slot {asked:?} for {}", self.callable.name),
                )
                .remedy(remedy));
            }
        };
        Ok(self.choices.entry(slot.clone()).or_insert_with(|| Choice {
            slot,
            ..Choice::default()
        }))
    }

    /// Folds a field name onto its declared spelling; an exact spelling wins.
    fn fold(&self, key: &str) -> Result<String, Error> {
        if self.callable.field(key).is_some() {
            return Ok(key.to_owned());
        }
        let matches: Vec<&Field> = self
            .callable
            .request
            .fields
            .iter()
            .filter(|f| f.name.eq_ignore_ascii_case(key))
            .collect();
        match matches.as_slice() {
            [] => Ok(key.to_owned()),
            [one] => Ok(one.name.clone()),
            _ => Err(invalid(
                "request_field_case_ambiguous",
                format!(
                    "{key:?} folds onto more than one of {}'s fields",
                    self.callable.name
                ),
            )
            .remedy("spell the field exactly")),
        }
    }

    /// A scalar spelled as the field's declared type.
    fn typed(&self, field: &Field, raw: &str) -> Result<Value, Error> {
        scalar(&field.ty, raw).ok_or_else(|| {
            let name = format!("{}.{}", self.callable.name, field.name);
            match &field.ty {
                Type::Asset(_) => invalid(
                    "asset_needs_file",
                    format!("{name} is an input file and `key=value` cannot attach one"),
                )
                .remedy(format!("use --asset {}=<file>", field.name)),
                Type::List(_) | Type::Map(_) | Type::Struct(_) | Type::Model => invalid(
                    "value_not_scalar",
                    format!("{name} is not a scalar and `key=value` cannot spell one"),
                )
                .remedy(format!("carry it as {}:=<json>", field.name)),
                ty => invalid(
                    "value_type",
                    format!(
                        "{name} is declared {} and {raw:?} is not one",
                        describe::phrase(ty)
                    ),
                )
                .remedy(format!("structured values are {}:=<json>", field.name)),
            }
        })
    }

    /// Whether a bare term is a local file for a media input: an explicit path (which must
    /// exist), or an existing file when the callable takes media. Never a `key=value` term.
    fn positional_file(&self, term: &str) -> Result<bool, Error> {
        let explicit = is_explicit_path(term) && !term.contains(char::is_whitespace);
        if !explicit && term.contains('=') || !self.takes_media() {
            return Ok(false);
        }
        let path = self.resolve(Path::new(term));
        if path.is_file() {
            return Ok(true);
        }
        if explicit {
            return Err(Error::new(
                Kind::NotFound,
                "input_file_absent",
                format!("input file {term} does not exist"),
            ));
        }
        Ok(false)
    }

    fn takes_media(&self) -> bool {
        self.callable.assets.is_some()
            || self
                .callable
                .request
                .fields
                .iter()
                .any(|f| f.ty.media().is_some())
    }

    /// An absolute path: `~` is the user's home, a relative path starts at the working directory.
    fn resolve(&self, path: &Path) -> PathBuf {
        let path = match path.strip_prefix("~") {
            Ok(rest) => self.terms.user_home.join(rest),
            Err(_) => path.to_path_buf(),
        };
        self.terms.cwd.join(path)
    }

    /// Binds every file: those named in the payload, `--asset` and positional files. Each value
    /// becomes the file's digest, and the file its InputFile.
    fn attach(&mut self) -> Result<Vec<InputFile>, Error> {
        let mut specs: Vec<Spec> = Vec::new();
        for field in &self.callable.request.fields {
            if let Some(value) = self.document.get(&field.name) {
                embedded_files(&field.ty, value, &field.name, &mut specs)?;
            }
        }
        for raw in &self.terms.assets {
            let spec = self.asset_spec(raw, &specs);
            specs.push(spec);
        }
        if let Some(assets) = &self.callable.assets {
            self.document
                .entry(assets.parameter.clone())
                .or_insert_with(|| Value::Array(vec![]));
        }
        let positional = std::mem::take(&mut self.files);
        self.bind_positional(&positional, &mut specs)?;
        self.preflight(&specs)?;

        let mut seen = BTreeSet::new();
        let mut files = Vec::new();
        for spec in specs {
            let parts = match spec.field {
                Some(field) => {
                    let mut parts: Vec<String> = field.split('.').map(str::to_owned).collect();
                    if parts.iter().any(|p| p.is_empty()) {
                        return Err(usage(
                            "asset_path_malformed",
                            format!("--asset field path {field:?} is malformed"),
                        ));
                    }
                    parts[0] = self.fold(&parts[0])?;
                    parts
                }
                None => {
                    let Some(assets) = &self.callable.assets else {
                        return Err(usage(
                            "assets_undeclared",
                            format!(
                                "{} declares no Assets input for {}",
                                self.callable.name,
                                spec.source.display()
                            ),
                        )
                        .remedy("use --asset <field-path>=<file> for a named input"));
                    };
                    let list =
                        self.document[&assets.parameter]
                            .as_array_mut()
                            .ok_or_else(|| {
                                invalid(
                                    "assets_not_list",
                                    format!("{} must be a list", assets.parameter),
                                )
                            })?;
                    let mut entry = Map::new();
                    if let Some(label) = &spec.label {
                        entry.insert("label".into(), label.clone().into());
                    }
                    list.push(Value::Object(entry));
                    vec![
                        assets.parameter.clone(),
                        (list.len() - 1).to_string(),
                        "asset".into(),
                    ]
                }
            };
            let field = parts.join(".");
            if !seen.insert(field.clone()) {
                return Err(invalid(
                    "asset_repeated",
                    format!("input file field {field:?} was given more than once"),
                ));
            }
            let accepts = self.accepts(&parts).ok_or_else(|| {
                invalid(
                    "asset_field_unknown",
                    format!(
                        "{}.{field} is not an input file field of this function",
                        self.callable.name
                    ),
                )
                .remedy(describe::usage_line(self.target, self.callable))
            })?;
            let path = self.resolve(&spec.source);
            let sniffed = sniff(&path)?;
            let (kind_max, kinds) = accepts;
            let max = kinds
                .iter()
                .find(|(types, _)| {
                    types
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&sniffed.media_type))
                })
                .or_else(|| kinds.iter().find(|(types, _)| types.is_empty()))
                .map(|(_, max)| max.unwrap_or(kind_max))
                .ok_or_else(|| {
                    let accepted: Vec<&str> = kinds
                        .iter()
                        .flat_map(|(t, _)| t.iter().map(String::as_str))
                        .collect();
                    invalid(
                        "asset_media_type",
                        format!(
                            "{field} accepts [{}], not {} ({})",
                            accepted.join(", "),
                            sniffed.media_type,
                            spec.source.display()
                        ),
                    )
                })?;
            if sniffed.length > max {
                return Err(invalid(
                    "input_asset_bound",
                    format!(
                        "{field} is {} bytes; it accepts at most {max}",
                        sniffed.length
                    ),
                ));
            }
            let digest = digest(&path)?;
            set_path(
                &mut self.document,
                &parts,
                Value::String(digest.to_string()),
            )?;
            let order = parts
                .iter()
                .filter_map(|p| p.parse::<u32>().ok())
                .next_back()
                .unwrap_or(0);
            files.push(InputFile {
                field,
                path,
                digest,
                length: sniffed.length,
                media_type: sniffed.media_type,
                order,
            });
        }
        self.count_kinds(&files)?;
        files.sort_by(|a, b| a.field.cmp(&b.field));
        Ok(files)
    }

    /// `field.path=file` names an input (a list of files named whole takes its next slot);
    /// `label=file` labels an Assets occurrence; a bare file joins the Assets collection. An
    /// explicit path is never split on `=`.
    fn asset_spec(&self, raw: &str, specs: &[Spec]) -> Spec {
        let bare = Spec {
            field: None,
            source: PathBuf::from(raw),
            label: None,
        };
        if is_explicit_path(raw) {
            return bare;
        }
        let Some((name, source)) = raw.split_once('=') else {
            return bare;
        };
        let source = PathBuf::from(source.trim());
        let name = name.trim();
        let parts: Vec<String> = name.split('.').map(str::to_owned).collect();
        let first = self.fold(&parts[0]).ok();
        let list = first.as_deref().filter(|first| {
            let field = self.callable.field(first);
            parts.len() == 1
                && field
                    .and_then(|f| f.ty.media())
                    .is_some_and(|(_, list)| list)
        });
        if let Some(list) = list {
            let held = self
                .document
                .get(list)
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            let prefix = format!("{list}.");
            let slot = |s: &Spec| -> Option<usize> {
                let rest = s.field.as_deref()?.strip_prefix(&prefix)?;
                rest.split('.').next()?.parse().ok()
            };
            let next = specs
                .iter()
                .filter_map(slot)
                .map(|i| i + 1)
                .max()
                .unwrap_or(0);
            return Spec {
                field: Some(format!("{list}.{}", held.max(next))),
                source,
                label: None,
            };
        }
        let named = self.callable.assets.is_none()
            || first.is_some_and(|first| {
                self.accepts(&[&[first][..], &parts[1..]].concat())
                    .is_some()
            });
        if named {
            Spec {
                field: Some(name.to_owned()),
                source,
                label: None,
            }
        } else {
            Spec {
                field: None,
                source,
                label: Some(name.to_owned()),
            }
        }
    }

    /// Each positional file fills the one free media input of its kind.
    fn bind_positional(&self, files: &[PathBuf], specs: &mut Vec<Spec>) -> Result<(), Error> {
        struct Input<'a> {
            name: &'a str,
            kind: &'a str,
            media_types: &'a [String],
            list: bool,
            assets: bool,
        }
        let mut inputs: Vec<Input> = Vec::new();
        if let Some(assets) = &self.callable.assets {
            for kind in &assets.kinds {
                inputs.push(Input {
                    name: &assets.parameter,
                    kind: &kind.kind,
                    media_types: &kind.media_types,
                    list: true,
                    assets: true,
                });
            }
        }
        for field in &self.callable.request.fields {
            if let (Some((kind, list)), false) =
                (field.ty.media(), self.callable.is_assets(&field.name))
            {
                inputs.push(Input {
                    name: &field.name,
                    kind,
                    media_types: &field.asset_bound.media_types,
                    list,
                    assets: false,
                });
            }
        }
        let mut taken: BTreeSet<String> = BTreeSet::new();
        let mut next: BTreeMap<String, usize> = BTreeMap::new();
        for (name, value) in &self.document {
            match value {
                Value::Array(list) => {
                    next.insert(name.clone(), list.len());
                }
                Value::Null => {}
                Value::String(s) if s.is_empty() => {}
                _ => {
                    taken.insert(name.clone());
                }
            }
        }
        for spec in specs.iter() {
            if let Some(field) = &spec.field {
                let mut parts = field.split('.');
                let first = self
                    .fold(parts.next().unwrap_or_default())
                    .unwrap_or_default();
                if let Some(index) = parts.next().and_then(|p| p.parse::<usize>().ok()) {
                    let at = next.entry(first.clone()).or_default();
                    *at = (*at).max(index + 1);
                }
                taken.insert(first);
            }
        }
        for file in files {
            let media_type = sniff(file)?.media_type;
            let rank = |input: &Input| -> u8 {
                let accepted = input.media_types.is_empty()
                    || input
                        .media_types
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&media_type));
                let category = media_type.split('/').next().unwrap_or_default();
                match (accepted, input.kind) {
                    (false, _) => 0,
                    (true, kind) if kind == category => 2,
                    (true, "file") => 1,
                    _ => 0,
                }
            };
            let best = inputs
                .iter()
                .filter(|i| i.list || !taken.contains(i.name))
                .map(rank)
                .max()
                .unwrap_or(0);
            let mut chosen: Vec<&Input> = inputs
                .iter()
                .filter(|i| (i.list || !taken.contains(i.name)) && best > 0 && rank(i) == best)
                .collect();
            chosen.dedup_by_key(|i| i.name);
            let shown = file.display();
            match chosen.as_slice() {
                [] => {
                    let names: Vec<String> = inputs
                        .iter()
                        .map(|i| format!("{} ({})", i.name, i.kind))
                        .collect();
                    return Err(invalid(
                        "positional_file_unbound",
                        format!(
                            "{shown} is {media_type}, which no free input of {} accepts",
                            self.callable.name
                        ),
                    )
                    .remedy(format!("its media inputs: {}", names.join(", "))));
                }
                [input] => {
                    let field = match (input.assets, input.list) {
                        (true, _) => None,
                        (false, true) => {
                            let at = next.entry(input.name.to_owned()).or_default();
                            *at += 1;
                            Some(format!("{}.{}", input.name, *at - 1))
                        }
                        (false, false) => {
                            taken.insert(input.name.to_owned());
                            Some(input.name.to_owned())
                        }
                    };
                    specs.push(Spec {
                        field,
                        source: file.clone(),
                        label: None,
                    });
                }
                several => {
                    let names: Vec<&str> = several.iter().map(|i| i.name).collect();
                    let choice = |i: &&Input| match i.assets {
                        true => format!("--asset {shown} (joins {})", i.name),
                        false => format!("--asset {}={shown}", i.name),
                    };
                    let choices: Vec<String> = several.iter().map(choice).collect();
                    return Err(usage(
                        "positional_file_ambiguous",
                        format!(
                            "{shown} is {media_type}, which inputs {} each accept",
                            names.join(" and ")
                        ),
                    )
                    .remedy(format!("name one: {}", choices.join(" or "))));
                }
            }
        }
        Ok(())
    }

    /// Labels are unique and the Assets collection holds no more than it declares, before any
    /// file is read.
    fn preflight(&self, specs: &[Spec]) -> Result<(), Error> {
        let Some(assets) = &self.callable.assets else {
            return Ok(());
        };
        let existing = self.document[&assets.parameter]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut labels = BTreeSet::new();
        let mut count = existing.len() as u64;
        let labelled = existing
            .iter()
            .filter_map(|e| e["label"].as_str().map(str::to_owned));
        for label in labelled.chain(
            specs
                .iter()
                .filter(|s| s.field.is_none())
                .filter_map(|s| s.label.clone()),
        ) {
            if !labels.insert(label.clone()) {
                return Err(invalid(
                    "asset_label_repeated",
                    format!("asset label {label:?} was given more than once"),
                ));
            }
        }
        count += specs.iter().filter(|s| s.field.is_none()).count() as u64;
        if let Some(max) = self
            .callable
            .field(&assets.parameter)
            .and_then(|f| f.constraints.max_length)
            && count > max
        {
            return Err(invalid(
                "input_asset_count",
                format!(
                    "{} has {count} files; it takes at most {max}",
                    assets.parameter
                ),
            ));
        }
        Ok(())
    }

    fn count_kinds(&self, files: &[InputFile]) -> Result<(), Error> {
        let Some(assets) = &self.callable.assets else {
            return Ok(());
        };
        for kind in &assets.kinds {
            let count = files
                .iter()
                .filter(|f| f.field.starts_with(&format!("{}.", assets.parameter)))
                .filter(|f| {
                    kind.media_types
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&f.media_type))
                })
                .count() as u64;
            if let Some(max) = kind.max_count.filter(|max| count > *max) {
                return Err(invalid(
                    "input_asset_count",
                    format!(
                        "{} has {count} {} files; it takes at most {max}",
                        assets.parameter, kind.kind
                    ),
                ));
            }
        }
        Ok(())
    }

    /// What the input at `parts` accepts: the default byte bound, and each media-type set with
    /// its own bound. None: no input file field is there.
    fn accepts(&self, parts: &[String]) -> Option<Accepts> {
        if let Some(AssetsSlot { parameter, kinds }) = &self.callable.assets
            && parts.len() == 3
            && &parts[0] == parameter
            && parts[2] == "asset"
            && parts[1].parse::<u32>().is_ok()
        {
            let kinds = kinds
                .iter()
                .map(|k| (k.media_types.clone(), k.max_bytes))
                .collect();
            return Some((DEFAULT_MAX_BYTES, kinds));
        }
        let field = self.callable.field(&parts[0])?;
        asset_at(
            &field.ty,
            &parts[1..],
            &field.asset_bound.media_types,
            field.asset_bound.max_bytes,
        )
        .map(|(types, max)| (max.unwrap_or(DEFAULT_MAX_BYTES), vec![(types, None)]))
    }

    /// An omitted integer `seed` comes from the idempotency key, so a retried key rebuilds the
    /// same payload. An `int | None` seed is the package's to choose.
    fn seed(&mut self) {
        let declared = self
            .callable
            .field("seed")
            .is_some_and(|f| f.ty == Type::Int);
        if declared && !self.document.contains_key("seed") {
            let digest = Sha256::digest(format!("cozy-seed\0{}", self.terms.key));
            let seed =
                u64::from_be_bytes(digest[..8].try_into().expect("8 bytes")) & ((1 << 53) - 1);
            self.document.insert("seed".into(), seed.into());
        }
    }
}

/// What an input file field accepts: its default byte bound, and each set of media types with
/// its own bound.
type Accepts = (u64, Vec<(Vec<String>, Option<u64>)>);

struct Spec {
    field: Option<String>,
    source: PathBuf,
    label: Option<String>,
}

/// The input-file spec at `parts` below a field of type `ty`: its media types and byte bound.
fn asset_at(
    ty: &Type,
    parts: &[String],
    types: &[String],
    max: Option<u64>,
) -> Option<(Vec<String>, Option<u64>)> {
    match ty {
        Type::Asset(_) if parts.is_empty() => Some((types.to_vec(), max)),
        Type::Union { branches, .. } => {
            branches.iter().find_map(|b| asset_at(b, parts, types, max))
        }
        Type::List(item) if !parts.is_empty() && parts[0].parse::<u32>().is_ok() => {
            asset_at(item, &parts[1..], types, max)
        }
        Type::Struct(s) if !parts.is_empty() => {
            let field = s.fields.iter().find(|f| f.name == parts[0])?;
            let (types, max) = if field.asset_bound.media_types.is_empty() {
                (types, field.asset_bound.max_bytes.or(max))
            } else {
                (
                    field.asset_bound.media_types.as_slice(),
                    field.asset_bound.max_bytes.or(max),
                )
            };
            asset_at(&field.ty, &parts[1..], types, max)
        }
        _ => None,
    }
}

/// Files named inside the payload: a string at an input-file position that is not a digest.
fn embedded_files(ty: &Type, value: &Value, path: &str, out: &mut Vec<Spec>) -> Result<(), Error> {
    match (ty, value) {
        (Type::Asset(_), Value::String(source))
            if !source.starts_with("sha256:") && !source.is_empty() =>
        {
            if source.contains("://") {
                return Err(invalid(
                    "asset_url",
                    format!("{path} needs a local file, not a URL"),
                ));
            }
            out.push(Spec {
                field: Some(path.to_owned()),
                source: source.into(),
                label: None,
            });
        }
        (Type::Union { branches, .. }, Value::String(_))
            if branches.iter().any(|b| {
                *b == Type::Str || scalar(b, value.as_str().unwrap_or_default()).is_some()
            }) => {}
        (Type::Union { branches, .. }, _) => {
            let before = out.len();
            for branch in branches {
                embedded_files(branch, value, path, out)?;
                if out.len() > before {
                    break;
                }
            }
        }
        (Type::List(item), Value::Array(values)) => {
            for (i, value) in values.iter().enumerate() {
                embedded_files(item, value, &format!("{path}.{i}"), out)?;
            }
        }
        (Type::Struct(s), Value::Object(object)) => {
            for field in &s.fields {
                if let Some(value) = object.get(&field.name) {
                    embedded_files(&field.ty, value, &format!("{path}.{}", field.name), out)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Rewrites relative file names at input-file positions against `base`.
fn relative_files(ty: &Type, value: &mut Value, base: &Path) {
    match (ty, value) {
        (Type::Asset(_), Value::String(source))
            if !source.contains("://")
                && !source.starts_with("sha256:")
                && !is_explicit_path(source)
                && !source.is_empty() =>
        {
            *source = base.join(&*source).to_string_lossy().into_owned();
        }
        (Type::Asset(_), Value::String(source))
            if source.starts_with("./") || source.starts_with("../") =>
        {
            *source = base.join(&*source).to_string_lossy().into_owned();
        }
        (Type::Union { branches, .. }, value) if !value.is_string() => {
            for branch in branches {
                relative_files(branch, value, base);
            }
        }
        (Type::List(item), Value::Array(values)) => values
            .iter_mut()
            .for_each(|v| relative_files(item, v, base)),
        (Type::Struct(s), Value::Object(object)) => {
            for field in &s.fields {
                if let Some(value) = object.get_mut(&field.name) {
                    relative_files(&field.ty, value, base);
                }
            }
        }
        _ => {}
    }
}

/// Writes `value` at `parts`, creating objects and list slots on the way.
fn set_path(
    document: &mut Map<String, Value>,
    parts: &[String],
    value: Value,
) -> Result<(), Error> {
    let mut node = document.entry(parts[0].clone()).or_insert(Value::Null);
    for part in &parts[1..] {
        if node.is_null() {
            *node = match part.parse::<usize>() {
                Ok(_) => Value::Array(vec![]),
                Err(_) => Value::Object(Map::new()),
            };
        }
        node = match node {
            Value::Array(list) => {
                let index: usize = part.parse().map_err(|_| {
                    invalid(
                        "asset_path_malformed",
                        format!("{part:?} is not a list position in {}", parts.join(".")),
                    )
                })?;
                if list.len() <= index {
                    list.resize(index + 1, Value::Null);
                }
                &mut list[index]
            }
            Value::Object(object) => object.entry(part.clone()).or_insert(Value::Null),
            _ => {
                return Err(invalid(
                    "asset_path_malformed",
                    format!(
                        "{} crosses a value that is not an object or list",
                        parts.join(".")
                    ),
                ));
            }
        };
    }
    *node = value;
    Ok(())
}

/// A scalar spelled for `ty`; None when the type has no such spelling.
pub(super) fn scalar(ty: &Type, raw: &str) -> Option<Value> {
    match ty {
        Type::Str | Type::Tree | Type::Unknown => Some(Value::String(raw.to_owned())),
        Type::Int => integer(raw),
        Type::Float => raw
            .parse::<f64>()
            .ok()
            .filter(|f| f.is_finite())
            .and_then(Number::from_f64)
            .map(Value::Number),
        Type::Bool => match raw {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            _ => None,
        },
        Type::Null => (raw == "null").then_some(Value::Null),
        Type::Literal(members) => {
            let text = Value::String(raw.to_owned());
            if members.contains(&text) {
                return Some(text);
            }
            let parsed: Value = serde_json::from_str(raw).ok()?;
            members.iter().any(|m| same(m, &parsed)).then_some(parsed)
        }
        Type::Union { branches, .. } => {
            let scalars = || {
                branches.iter().filter(|b| {
                    matches!(
                        b,
                        Type::Str
                            | Type::Int
                            | Type::Float
                            | Type::Bool
                            | Type::Null
                            | Type::Literal(_)
                    )
                })
            };
            // A branch with an unambiguous JSON spelling wins over a string branch.
            if let Ok(parsed) = serde_json::from_str::<Value>(raw)
                && !parsed.is_string()
                && scalars().any(|b| check(b, &mut parsed.clone(), "", &mut vec![]).is_ok())
            {
                return Some(match parsed {
                    Value::Number(_) => scalars()
                        .find_map(|b| (*b != Type::Str).then(|| scalar(b, raw)).flatten())
                        .unwrap_or(parsed),
                    other => other,
                });
            }
            scalars()
                .filter(|b| matches!(b, Type::Str | Type::Literal(_)))
                .find_map(|b| scalar(b, raw))
        }
        Type::List(_) | Type::Map(_) | Type::Struct(_) | Type::Asset(_) | Type::Model => None,
    }
}

fn integer(raw: &str) -> Option<Value> {
    raw.parse::<i64>()
        .map(Value::from)
        .or_else(|_| raw.parse::<u64>().map(Value::from))
        .ok()
}

/// JSON values equal as the interface means them: `1` and `1.0` are one number.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

/// Validates the payload against the request, dropping undeclared fields at any depth. Answers
/// the dropped paths; refuses with every problem at once.
fn validate(
    target: &str,
    callable: &Callable,
    document: &mut Map<String, Value>,
) -> Result<Vec<String>, Error> {
    let mut ignored = Vec::new();
    let mut problems = Vec::new();
    let mut missing = Vec::new();
    document.retain(|name, value| {
        if callable.field(name).is_some() {
            return true;
        }
        if name == LORA_MODE {
            if !matches!(value.as_str(), Some("bake" | "per_step")) {
                problems.push(format!("{LORA_MODE} must be \"bake\" or \"per_step\""));
            }
            return true;
        }
        ignored.push(name.clone());
        false
    });
    for field in &callable.request.fields {
        match document.get_mut(&field.name) {
            None if field.wire == Wire::Required => missing.push(describe::signature(field)),
            None => {}
            Some(value) => {
                if let Err(problem) = check_field(field, value, &field.name, &mut ignored) {
                    problems.push(problem);
                }
            }
        }
    }
    ignored.sort();
    if !missing.is_empty() {
        problems.insert(
            0,
            format!("provide required arguments: [{}]", missing.join("; ")),
        );
    }
    if problems.is_empty() {
        return Ok(ignored);
    }
    Err(invalid(
        "request_payload_invalid",
        format!(
            "{target} request payload is invalid: {}",
            problems.join("; ")
        ),
    )
    .remedy(describe::usage_line(target, callable)))
}

fn check_field(
    field: &Field,
    value: &mut Value,
    path: &str,
    ignored: &mut Vec<String>,
) -> Result<(), String> {
    check(&field.ty, value, path, ignored)?;
    bounds(&field.constraints, value, path)
}

/// Checks `value` against `ty`, dropping undeclared struct members into `ignored`.
fn check(
    ty: &Type,
    value: &mut Value,
    path: &str,
    ignored: &mut Vec<String>,
) -> Result<(), String> {
    let ok = match (ty, &mut *value) {
        (Type::Unknown | Type::Model, _) => true,
        (Type::Str, Value::String(_))
        | (Type::Bool, Value::Bool(_))
        | (Type::Null, Value::Null) => true,
        (Type::Int, Value::Number(n)) => n.is_i64() || n.is_u64(),
        (Type::Float, Value::Number(_)) => true,
        (Type::Literal(members), value) => members.iter().any(|m| same(m, value)),
        (Type::Asset(_) | Type::Tree, Value::String(s)) => !s.is_empty(),
        (
            Type::Union {
                branches,
                tag_field,
            },
            value,
        ) => {
            for branch in branches {
                let mut tried = value.clone();
                let mut dropped = Vec::new();
                let branch = match (branch, tag_field) {
                    (Type::Struct(s), Some(tag)) if s.tag_field.is_none() => {
                        Type::Struct(Box::new(Struct {
                            tag_field: Some(tag.clone()),
                            ..(**s).clone()
                        }))
                    }
                    _ => branch.clone(),
                };
                if check(&branch, &mut tried, path, &mut dropped).is_ok() {
                    *value = tried;
                    ignored.extend(dropped);
                    return Ok(());
                }
            }
            return Err(format!("{path} matches none of {}", describe::phrase(ty)));
        }
        (Type::List(item), Value::Array(values)) => {
            for (i, value) in values.iter_mut().enumerate() {
                check(item, value, &format!("{path}[{i}]"), ignored)?;
            }
            true
        }
        (Type::Map(item), Value::Object(values)) => {
            for (key, value) in values.iter_mut() {
                check(item, value, &format!("{path}.{key}"), ignored)?;
            }
            true
        }
        (Type::Struct(s), Value::Object(object)) => {
            if let Some(tag) = &s.tag_field
                && s.tag.is_some()
                && object.get(tag) != s.tag.as_ref()
            {
                return Err(format!("{path} has an absent or other {tag}"));
            }
            object.retain(|name, _| {
                let keep = s.fields.iter().any(|f| &f.name == name)
                    || s.tag_field.as_deref() == Some(name);
                if !keep {
                    ignored.push(format!("{path}.{name}"));
                }
                keep
            });
            for field in &s.fields {
                match object.get_mut(&field.name) {
                    Some(value) => {
                        check_field(field, value, &format!("{path}.{}", field.name), ignored)?
                    }
                    None if field.wire == Wire::Required => {
                        return Err(format!("{path} omits required {}", field.name));
                    }
                    None => {}
                }
            }
            true
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{path} is not {}", describe::phrase(ty)))
    }
}

fn bounds(c: &Constraints, value: &Value, path: &str) -> Result<(), String> {
    let length = match value {
        Value::String(s) => Some(s.chars().count() as u64),
        Value::Array(a) => Some(a.len() as u64),
        Value::Object(o) => Some(o.len() as u64),
        _ => None,
    };
    if let Some(length) = length {
        if let Some(min) = c.min_length.filter(|min| length < *min) {
            return Err(format!("{path} has length {length}; its minimum is {min}"));
        }
        if let Some(max) = c.max_length.filter(|max| length > *max) {
            return Err(format!("{path} has length {length}; its maximum is {max}"));
        }
    }
    let Some(number) = value.as_f64() else {
        return Ok(());
    };
    let shown = value.to_string();
    for (bound, holds, says) in [
        (c.ge, number >= c.ge.unwrap_or_default(), "its minimum is"),
        (
            c.gt,
            number > c.gt.unwrap_or_default(),
            "it must be greater than",
        ),
        (c.le, number <= c.le.unwrap_or_default(), "its maximum is"),
        (
            c.lt,
            number < c.lt.unwrap_or_default(),
            "it must be less than",
        ),
    ] {
        if let Some(bound) = bound.filter(|_| !holds) {
            return Err(format!("{path} is {shown}; {says} {bound}"));
        }
    }
    if let Some(step) = c.multiple_of.filter(|s| *s != 0.0) {
        let ratio = number / step;
        if (ratio - ratio.round()).abs() > 1e-9 {
            return Err(format!(
                "{path} is {shown}; it must be a multiple of {step}"
            ));
        }
    }
    Ok(())
}

struct Sniffed {
    length: u64,
    media_type: String,
}

/// A file's size and media type, from its first bytes.
fn sniff(path: &Path) -> Result<Sniffed, Error> {
    let unreadable = |e: std::io::Error| {
        Error::new(
            Kind::NotFound,
            "input_file_unreadable",
            format!("input file {}: {e}", path.display()),
        )
    };
    let mut file = File::open(path).map_err(unreadable)?;
    let length = file.metadata().map_err(unreadable)?.len();
    let mut head = Vec::with_capacity(8192);
    file.by_ref()
        .take(8192)
        .read_to_end(&mut head)
        .map_err(unreadable)?;
    let media_type = match infer::get(&head).map(|t| t.mime_type()) {
        Some("audio/x-wav" | "audio/wave") => "audio/wav",
        Some("audio/x-flac") => "audio/flac",
        Some(other) => other,
        None => "application/octet-stream",
    };
    Ok(Sniffed {
        length,
        media_type: media_type.to_owned(),
    })
}

fn digest(path: &Path) -> Result<Digest, Error> {
    let mut file = File::open(path).map_err(|e| {
        Error::new(
            Kind::NotFound,
            "input_file_unreadable",
            format!("input file {}: {e}", path.display()),
        )
    })?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| {
        Error::new(
            Kind::Internal,
            "input_file_unreadable",
            format!("input file {}: {e}", path.display()),
        )
    })?;
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

/// A LoRA strength as a canonical decimal string: finite, `0` for zero.
fn canonical_scale(raw: &str) -> Result<String, Error> {
    let malformed = || {
        usage(
            "lora_strength_invalid",
            format!("LoRA strength {raw:?} must be a finite decimal number"),
        )
    };
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || !trimmed
            .chars()
            .all(|c| c.is_ascii_digit() || "+-.eE".contains(c))
    {
        return Err(malformed());
    }
    let value: f64 = trimmed.parse().map_err(|_| malformed())?;
    if !value.is_finite() {
        return Err(malformed());
    }
    Ok(if value == 0.0 {
        "0".into()
    } else {
        format!("{value}")
    })
}

fn nonempty(reference: &str, slot: &str) -> Result<String, Error> {
    let reference = reference.trim();
    if reference.is_empty() {
        return Err(usage(
            "lora_malformed",
            format!("a LoRA for {slot} names no reference"),
        ));
    }
    Ok(reference.to_owned())
}

/// `[org/pkg/]callable.models.param`: another callable's slot, which its machine checks.
fn child_slot(asked: &str) -> bool {
    let tail = asked.rsplit('/').next().unwrap_or(asked);
    let mut parts = tail.split('.');
    let ident = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some(callable), Some("models"), Some(param), None) if ident(callable) && ident(param)
    ) && asked.matches('/').count() != 1
}

pub(super) fn is_explicit_path(raw: &str) -> bool {
    ["/", "~/", "./", "../"].iter().any(|p| raw.starts_with(p)) || raw == "~"
}

fn usage(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Usage, code, message)
}

fn invalid(code: &str, message: impl Into<String>) -> Error {
    Error::new(Kind::Validation, code, message)
}

#[cfg(test)]
#[path = "bind_tests.rs"]
mod tests;
