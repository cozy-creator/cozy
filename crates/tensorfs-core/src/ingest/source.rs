//! Reviewed foreign-source planning and preparation shared by the CLI and Python facade.
//!
//! The network and its credentials end before this module. Inputs are verified immutable local
//! paths plus exact provider identities. TensorFS re-fences those paths, re-proves the reviewed
//! header/profile match, streams selected tensor bytes through the ordinary ingest/CAS pass, and
//! retains only exact CAS holds and typed SQLite result rows.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::canon::{self, Value};
use crate::durability;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{tensor_schema_digest_of, tensor_schema_value};
use crate::ids::{ascii_name, object_id, prefixed, ObjectRef};
use crate::registry;
use crate::spec::EncodingSpec;
use crate::store::Store;

use super::carrier::{self, SourceHeader};
use super::convert::{self, Converter, Plan, Rekey, Source, Target};
use super::fingerprint::{
    self, Banked, Fingerprint, FingerprintRegistry, Grammar, SourceProfile, SourceProfileComponent,
};
use super::transaction::{self, SourceFile};

pub const BUILTIN_REGISTRY: &str = "builtin:fingerprint-registry/1";
/// The profile of a source no reviewed profile matches: its carriers, stored as they are
/// ([`convert::IDENTITY`]). The registry is a hint that selects better; its absence is not
/// a refusal.
pub const AS_IS: &str = "as-is/1";
pub const BUILTIN_REGISTRY_BYTES: &[u8] =
    include_bytes!("../../../../vectors/fingerprints/registry.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarrierInput {
    pub member: Option<String>,
    pub path: PathBuf,
}

/// The carriers one plan may read, and the ONLY map from a member to a file.
///
/// **A path is an OBJECT's identity, never a MEMBER's and never a layout.** A fetched
/// source file lives at `<root>/staging/<sha256>`: the name is a hex digest, it carries no
/// extension, and the area is flat, so it has no siblings that mean anything. A path cannot
/// say what a carrier is, and a sharded index's `weight_map` cannot be resolved beside it.
/// Here the selection is the directory.
///
/// Member -> path, not path -> member: the map is deliberately one-way because it is not
/// injective. Two members carrying identical bytes are ONE object at ONE path (H3 ships
/// `FL2VA/transformer/model.safetensors.index.json` and its `Ref2VA/` counterpart as one
/// 38,323-byte object), so 48 members present 47 paths — and reading that shared object
/// under its two member names must, and does, yield two different shard sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CarrierSet {
    by_member: BTreeMap<String, PathBuf>,
}

impl CarrierSet {
    /// Every carrier that named itself. A carrier with no member contributes nothing: it
    /// is a location with no name, and nothing can name it back.
    pub fn of(carriers: &[CarrierInput]) -> Self {
        let mut by_member = BTreeMap::new();
        for carrier in carriers {
            if let Some(member) = &carrier.member {
                by_member.insert(member.clone(), carrier.path.clone());
            }
        }
        Self { by_member }
    }

    pub fn path_of(&self, member: &str) -> Option<&Path> {
        self.by_member.get(member).map(PathBuf::as_path)
    }

    pub fn members(&self) -> &BTreeMap<String, PathBuf> {
        &self.by_member
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSource {
    pub component: String,
    pub path: PathBuf,
    pub source_member: Option<String>,
    pub projected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePlan {
    pub carriers: Vec<CarrierInput>,
    pub construction_order: Vec<(String, String)>,
    pub converter: String,
    pub profile: String,
    pub registry: String,
    pub registry_sha256: String,
    pub session: String,
    pub sources: Vec<PlannedSource>,
    pub target: String,
}

impl SourcePlan {
    pub fn value(&self) -> Value {
        Value::obj(vec![
            (
                "carriers",
                Value::arr(
                    self.carriers
                        .iter()
                        .map(|carrier| {
                            let mut fields = vec![(
                                "path",
                                Value::str(carrier.path.to_string_lossy().to_string()),
                            )];
                            if let Some(member) = &carrier.member {
                                fields.push(("member", Value::str(member.clone())));
                            }
                            Value::obj(fields)
                        })
                        .collect(),
                ),
            ),
            (
                "construction_order",
                Value::arr(
                    self.construction_order
                        .iter()
                        .map(|(component, key)| {
                            Value::arr(vec![Value::str(component.clone()), Value::str(key.clone())])
                        })
                        .collect(),
                ),
            ),
            ("converter", Value::str(self.converter.clone())),
            ("profile", Value::str(self.profile.clone())),
            ("registry", Value::str(self.registry.clone())),
            ("registry_sha256", Value::str(self.registry_sha256.clone())),
            ("session", Value::str(self.session.clone())),
            (
                "sources",
                Value::arr(
                    self.sources
                        .iter()
                        .map(|source| {
                            let mut fields = vec![
                                ("component", Value::str(source.component.clone())),
                                (
                                    "path",
                                    Value::str(source.path.to_string_lossy().to_string()),
                                ),
                                ("projected", Value::Bool(source.projected)),
                            ];
                            if let Some(member) = &source.source_member {
                                fields.push(("source_member", Value::str(member.clone())));
                            }
                            Value::obj(fields)
                        })
                        .collect(),
                ),
            ),
            ("target", Value::str(self.target.clone())),
        ])
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canon::write(&self.value())
    }
}

pub struct Prepared {
    pub plan: Plan,
    pub conv: &'static Converter,
    pub seeds: Vec<(String, registry::Seed)>,
    pub files: Vec<SourceFile>,
    pub raws: Vec<Vec<u8>>,
    pub verdicts: Vec<Banked>,
    pub fps: Vec<Fingerprint>,
    pub target: Target,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CarrierUse {
    Unused,
    Whole,
    Projected,
}

#[derive(Clone)]
struct SourceAssignment {
    component: String,
    carrier: usize,
    projected: bool,
    variant: usize,
}

fn enumerate_assignments(
    index: usize,
    candidates: &[Vec<(usize, bool, usize)>],
    uses: &mut [CarrierUse],
    current: &mut Vec<SourceAssignment>,
    components: &[SourceProfileComponent],
    solutions: &mut Vec<Vec<SourceAssignment>>,
) {
    if solutions.len() >= 2 {
        return;
    }
    if index == candidates.len() {
        solutions.push(current.clone());
        return;
    }
    for (carrier, projected, variant) in &candidates[index] {
        let prior = uses[*carrier];
        let allowed = if *projected {
            matches!(prior, CarrierUse::Unused | CarrierUse::Projected)
        } else {
            prior == CarrierUse::Unused
        };
        if !allowed {
            continue;
        }
        uses[*carrier] = if *projected {
            CarrierUse::Projected
        } else {
            CarrierUse::Whole
        };
        current.push(SourceAssignment {
            component: components[index].component.clone(),
            carrier: *carrier,
            projected: *projected,
            variant: *variant,
        });
        enumerate_assignments(index + 1, candidates, uses, current, components, solutions);
        current.pop();
        uses[*carrier] = prior;
    }
}

/// Every assignment of carriers to the profile's components, and the first grammar refusal,
/// which explains an explicitly requested profile that matched nothing.
fn profile_assignments(
    profile: &SourceProfile,
    registry: &FingerprintRegistry,
    carriers: &[CarrierInput],
    headers: &[SourceHeader],
) -> (Vec<Vec<SourceAssignment>>, Option<Refusal>) {
    let mut candidates = Vec::new();
    let mut rejected = None;
    for component in &profile.components {
        let member_matches = |index: usize| match (
            component.source_member.as_deref(),
            component.source_member_prefix.as_deref(),
            carriers[index].member.as_deref(),
        ) {
            (Some(expected), None, Some(observed)) => expected == observed,
            (None, Some(prefix), Some(observed)) => observed.starts_with(prefix),
            // An explicit empty prefix permits this reviewed structure
            // under any filename, including a local unlabeled carrier.
            (None, Some(""), None) | (None, None, None) => true,
            _ => false,
        };
        // Carriers banked from this exact tensor schema first; otherwise any carrier with
        // the reviewed key set, whose shapes and dtypes the conversion itself validates.
        let mut exact = Vec::new();
        let mut hits = Vec::new();
        if let Some(grammar) = &profile.grammar {
            for index in (0..headers.len()).filter(|index| member_matches(*index)) {
                match grammar_match(component, grammar, &headers[index]) {
                    Ok(()) => hits.push((index, false, 0)),
                    Err(refusal) => drop(rejected.get_or_insert(refusal)),
                }
            }
        }
        for (variant_index, variant) in component.variants.iter().enumerate() {
            for entry in registry.entries.iter().filter(|entry| {
                entry.component == component.component
                    && entry.keyset_digest == variant.keyset_digest
            }) {
                let projector = convert::converter(&entry.converter)
                    .ok()
                    .filter(|c| c.dialect == "safetensors.single_file");
                let projected = projector.is_some();
                for (index, header) in headers.iter().enumerate() {
                    if !member_matches(index) {
                        continue;
                    }
                    let view = match projector {
                        Some(conv) => {
                            match convert::single_file_component(conv, &component.component, header)
                            {
                                Ok(view) => view,
                                Err(_) => continue,
                            }
                        }
                        None => header.clone(),
                    };
                    let observed = match fingerprint::fingerprint(&component.component, &view) {
                        Ok(observed) => observed,
                        Err(_) => continue,
                    };
                    if observed.keyset_digest != variant.keyset_digest {
                        continue;
                    }
                    let hit = (index, projected, variant_index);
                    if observed.tensor_schema_digest == entry.tensor_schema_digest
                        && observed.logical_keys as u64 == entry.logical_keys
                    {
                        exact.push(hit);
                    } else if !hits.contains(&hit) {
                        hits.push(hit);
                    }
                }
            }
        }
        let hits = if exact.is_empty() { hits } else { exact };
        if hits.is_empty() {
            return (Vec::new(), rejected);
        }
        candidates.push(hits);
    }
    let mut solutions = Vec::new();
    enumerate_assignments(
        0,
        &candidates,
        &mut vec![CarrierUse::Unused; headers.len()],
        &mut Vec::new(),
        &profile.components,
        &mut solutions,
    );
    (solutions, rejected)
}

/// A grammar matches when its reviewed converter plans the carrier as this component and
/// every key carries its prefix. The planner refuses unknown, incomplete or misshapen keys,
/// so the grammar is the converter's, stated once.
fn grammar_match(
    component: &SourceProfileComponent,
    grammar: &Grammar,
    header: &SourceHeader,
) -> Result<()> {
    let name = component.component.clone();
    let target = Target {
        components: vec![(name.clone(), component.target_encoding.clone())],
    };
    let source = Source::Carrier {
        component: name,
        file: 0,
        header,
    };
    let specs = plain_specs(&specs_for(&target, None)?);
    convert::plan(
        convert::converter(&grammar.converter)?,
        &[source],
        &target,
        &specs,
        &Rekey::default(),
    )?;
    let prefix = &grammar.key_prefix;
    match header.tensors.iter().find(|t| !t.key.starts_with(prefix)) {
        Some(tensor) => refuse(
            Code::KEY_GRAMMAR,
            format!(
                "{}: outside this profile's key prefix {prefix:?}",
                tensor.key
            ),
        ),
        None => Ok(()),
    }
}

/// Where a Store keeps a reviewed registry newer than the built-in one: shipped as a pinned
/// file or served by the Hub, and bound with [`bind_registry`]. It only extends the built-in
/// registry, so a TensorFS older than a model can still ingest it.
pub const REGISTRY_BINDING: &str = "fingerprint-registry.json";

/// Bind (or with `None`, clear) the Store's reviewed registry. The data is validated here,
/// provenance included, so a malformed file is a caller's mistake at bind time.
pub fn bind_registry(store: &Store, bytes: Option<&[u8]>) -> Result<()> {
    let path = store.root().join(REGISTRY_BINDING);
    let Some(bytes) = bytes else {
        return match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => refuse(Code::IO_FAILED, format!("clear registry binding: {error}")),
        };
    };
    FingerprintRegistry::parse(bytes)?;
    let temporary = store.root().join("tmp").join(format!(
        "{REGISTRY_BINDING}.{}",
        crate::meta::now_nanos_unique()
    ));
    let written = fs::write(&temporary, bytes)
        .and_then(|()| fs::File::open(&temporary)?.sync_all())
        .and_then(|()| fs::rename(&temporary, &path));
    if let Err(error) = written {
        let _ = fs::remove_file(&temporary);
        return refuse(Code::IO_FAILED, format!("bind registry: {error}"));
    }
    Ok(())
}

/// The converters a reviewed profile's components bind to, from registry data alone, so an
/// uploader keys its memo and fetches each converter's reference before reading a header.
pub fn profile_converters(registry: &[u8], profile: &str) -> Result<Vec<&'static Converter>> {
    if profile == AS_IS {
        return Ok(vec![convert::converter(convert::IDENTITY)?]);
    }
    let registry = FingerprintRegistry::parse(registry)?;
    let Some(found) = registry.source_profiles.iter().find(|p| p.name == profile) else {
        return refuse(
            Code::UNREGISTERED_FINGERPRINT,
            format!("reviewed source profile {profile:?} is absent"),
        );
    };
    let mut out: Vec<&'static Converter> = Vec::new();
    for component in &found.components {
        for variant in &component.variants {
            for entry in registry.entries.iter().filter(|e| {
                e.component == component.component && e.keyset_digest == variant.keyset_digest
            }) {
                let conv = convert::converter(&entry.converter)?;
                if !out.iter().any(|c| c.name == conv.name) {
                    out.push(conv);
                }
            }
        }
    }
    if let Some(grammar) = &found.grammar {
        out.push(convert::converter(&grammar.converter)?);
    }
    if out.is_empty() {
        return refuse(
            Code::UNREGISTERED_FINGERPRINT,
            format!("source profile {profile:?} binds no reviewed converter"),
        );
    }
    out.sort_by_key(|c| c.name);
    Ok(out)
}

/// The registry an ingest on this Store classifies with: the built-in one, extended by the
/// Store's bound registry, then by a caller-`supplied` one. A bound file that no longer
/// parses is skipped (it can only add); supplied data that does not parse refuses.
pub fn effective_registry(store: &Store, supplied: Option<&[u8]>) -> Result<(String, Vec<u8>)> {
    let mut registry = FingerprintRegistry::parse(BUILTIN_REGISTRY_BYTES)?;
    let mut locator = BUILTIN_REGISTRY.to_string();
    let bound = fs::read(store.root().join(REGISTRY_BINDING))
        .ok()
        .filter(|bytes| bytes.len() <= crate::limits::DOC_MAX_BYTES)
        .and_then(|bytes| FingerprintRegistry::parse(&bytes).ok());
    if let Some(bound) = bound {
        registry.extend(bound);
        locator.push_str("+store");
    }
    if let Some(bytes) = supplied {
        registry.extend(FingerprintRegistry::parse(bytes)?);
        locator.push_str("+supplied");
    }
    Ok((locator, registry.to_bytes()))
}

pub fn read_header_only(path: &Path) -> Result<(SourceHeader, Vec<u8>)> {
    let json = fs::read(path).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {}: {error}", path.display()),
    })?;
    let data_start = json.len() as u64 + 8;
    let header = carrier::parse_header(&json, data_start, None)?;
    let mut raw = (json.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(&json);
    Ok((header, raw))
}

/// Read one carrier's header, by NAME.
///
/// The reviewed member is the name; the file name is the name only when no member was
/// given. This is not a preference, it is the difference between working and not: a fetched
/// carrier is staged at `<root>/staging/<sha256>`, so a
/// `*.safetensors.index.json` arrives with a bare hex digest for a file name, falls
/// through every extension test, and is read as a safetensors file — whereupon its first
/// eight bytes, `{\n  "met`, are taken for a u64 header length of 8,387,229,874,382,703,227
/// and refused against the 16 MiB cap. That is H3 run 309, after 48/48 members and 210.3 GB.
///
/// The same key then decides where the shards are: a carrier read by member resolves them
/// through `set`, a carrier read by file name resolves them beside itself. One key, both
/// questions.
pub fn read_carrier(carrier: &CarrierInput, set: &CarrierSet) -> Result<(SourceHeader, Vec<u8>)> {
    let path = carrier.path.as_path();
    let name = match &carrier.member {
        Some(member) => member.as_str(),
        None => path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(""),
    };
    if name.ends_with(".index.json") {
        let shards = match &carrier.member {
            Some(member) => carrier::Shards::Selection {
                index: member,
                set: set.members(),
            },
            None => carrier::Shards::Siblings,
        };
        return carrier::read_sharded(path, shards);
    }
    if name.ends_with(".json") {
        return read_header_only(path);
    }
    let header = carrier::read_header(path)?;
    let mut file = fs::File::open(path).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("reopen {}: {error}", path.display()),
    })?;
    let mut raw = vec![0u8; header.header_bytes as usize + 8];
    file.read_exact(&mut raw).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("reread header of {}: {error}", path.display()),
    })?;
    Ok((header, raw))
}

/// A carrier named only by where it is: `tfs ingest --source <component>=<path>` over a real
/// directory tree. The file name is the name and the directory is the shard scope, which is
/// honest ONLY where a directory exists — which a Store object path is not.
pub fn read_carrier_at(path: &Path) -> Result<(SourceHeader, Vec<u8>)> {
    read_carrier(
        &CarrierInput {
            member: None,
            path: path.to_path_buf(),
        },
        &CarrierSet::default(),
    )
}

pub fn source_digest(header: &SourceHeader, raw_header: &[u8]) -> String {
    object_id(&canon::write(&Value::obj(vec![
        (
            "carrier_header_sha256",
            Value::str(crate::sha256::hex_digest(raw_header)),
        ),
        ("data_start", Value::uint(header.data_start)),
        ("file_len", Value::uint(header.file_len)),
        ("tensors", Value::uint(header.tensors.len() as u64)),
    ])))
}

fn specs_of(alias: &str, variant: Option<usize>) -> Result<Vec<registry::Seed>> {
    let hits: Vec<registry::Seed> = registry::seeds()
        .into_iter()
        .filter(|seed| seed.alias == alias)
        .collect();
    match (hits.len(), variant) {
        (0, _) => refuse(
            Code::UNKNOWN_ENCODING,
            format!("{alias:?} is not a platform alias"),
        ),
        (n, Some(index)) if index < n => Ok(vec![hits.into_iter().nth(index).unwrap()]),
        (n, Some(index)) => refuse(
            Code::UNKNOWN_ENCODING,
            format!("{alias:?} groups {n} specs; variant {index} names none"),
        ),
        (_, None) => Ok(hits),
    }
}

pub fn specs_for(target: &Target, variant: Option<usize>) -> Result<Vec<(String, registry::Seed)>> {
    let mut out: Vec<(String, registry::Seed)> = specs_of("plain/1", None)?
        .into_iter()
        .map(|seed| ("plain/1".to_string(), seed))
        .collect();
    for (_, alias) in &target.components {
        if out.iter().any(|(present, _)| present == alias) {
            continue;
        }
        out.extend(
            specs_of(alias, variant)?
                .into_iter()
                .map(|seed| (alias.clone(), seed)),
        );
    }
    Ok(out)
}

pub fn plain_specs(seeds: &[(String, registry::Seed)]) -> Vec<(String, EncodingSpec)> {
    seeds
        .iter()
        .map(|(alias, seed)| (alias.clone(), seed.spec.clone()))
        .collect()
}

pub fn plan_tensor_schema(plan: &Plan) -> Value {
    tensor_schema_value(plan.ops.iter().map(|operation| {
        (
            operation.component.as_str(),
            operation.out_key.as_str(),
            operation.logical_dtype,
            operation.logical_shape.as_slice(),
        )
    }))
}

/// How [`prepare`] names each source's converter.
#[derive(Clone, Copy)]
pub enum Classify<'a> {
    /// Every source stored as it is, through [`convert::IDENTITY`].
    AsIs,
    /// Each source's banked fingerprint names its converter.
    Banked(&'a FingerprintRegistry),
    /// A grammar profile matched: its reviewed converter plans every source.
    Grammar(&'static Converter),
}

/// How sources planned under the reviewed `profile` classify: a grammar profile's converter
/// plans them; any other profile authorizes each by its banked fingerprint.
pub fn classify<'a>(registry: &'a FingerprintRegistry, profile: &str) -> Result<Classify<'a>> {
    if profile == AS_IS {
        return Ok(Classify::AsIs);
    }
    let mut profiles = registry.source_profiles.iter();
    let found = profiles.find(|found| found.name == profile);
    Ok(match found.and_then(|found| found.grammar.as_ref()) {
        Some(grammar) => Classify::Grammar(convert::converter(&grammar.converter)?),
        None => Classify::Banked(registry),
    })
}

/// `sources` carries each component's CARRIER, not its path: a path alone cannot say what
/// the file is, and this is the second place — after the plan's own header pass — that
/// would otherwise read a CAS-resident index as a safetensors file.
pub fn prepare(
    target: &str,
    sources: &[(String, CarrierInput)],
    set: &CarrierSet,
    classify: Classify<'_>,
    spec_variant: Option<usize>,
) -> Result<Prepared> {
    let target = Target::parse(target)?;
    let seeds = specs_for(&target, spec_variant)?;
    let specs = plain_specs(&seeds);
    let (mut files, mut raws, mut verdicts, mut fingerprints) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (component, carrier) in sources {
        let path = &carrier.path;
        let (full, raw) = read_carrier(carrier, set)?;
        let Classify::Banked(registry) = classify else {
            raws.push(raw);
            files.push(SourceFile {
                path: path.clone(),
                header: full,
            });
            continue;
        };
        // A single-file converter's view of the carrier authorizes only under that converter.
        let mut selected = None;
        for conv in convert::CONVERTERS
            .iter()
            .filter(|c| c.dialect == "safetensors.single_file")
        {
            let Ok(view) = convert::single_file_component(conv, component, &full) else {
                continue;
            };
            let Ok(observed) = fingerprint::fingerprint(component, &view) else {
                continue;
            };
            if let Ok(banked) = registry.authorize(component, &observed) {
                if banked.converter == conv.name {
                    selected = Some((view, observed, banked));
                    break;
                }
            }
        }
        let (header, observed, banked) = match selected {
            Some(selected) => selected,
            None => {
                let observed = fingerprint::fingerprint(component, &full)?;
                let banked = registry.authorize(component, &observed)?;
                (full, observed, banked)
            }
        };
        verdicts.push(banked);
        fingerprints.push(observed);
        raws.push(raw);
        files.push(SourceFile {
            path: path.clone(),
            header,
        });
    }
    let names: Vec<&str> = verdicts
        .iter()
        .map(|verdict| verdict.converter.as_str())
        .collect();
    if names.windows(2).any(|window| window[0] != window[1]) {
        return refuse(
            Code::AMBIGUOUS_CLASSIFICATION,
            format!("reviewed components name different converters {names:?}"),
        );
    }
    let conv = match classify {
        Classify::Grammar(conv) => conv,
        _ => convert::converter(names.first().copied().unwrap_or(convert::IDENTITY))?,
    };
    let views: Vec<Source<'_>> = files
        .iter()
        .enumerate()
        .map(|(index, source)| Source::Carrier {
            component: sources[index].0.clone(),
            file: index,
            header: &source.header,
        })
        .collect();
    let plan = convert::plan(conv, &views, &target, &specs, &Rekey::default())?;
    drop(views);
    Ok(Prepared {
        plan,
        conv,
        seeds,
        files,
        raws,
        verdicts,
        fps: fingerprints,
        target,
    })
}

pub fn session_of(prepared: &Prepared) -> String {
    let value = Value::obj(vec![
        ("converter", Value::str(prepared.plan.converter.clone())),
        (
            "sources",
            Value::arr(
                prepared
                    .files
                    .iter()
                    .zip(prepared.raws.iter())
                    .map(|(source, raw)| Value::str(source_digest(&source.header, raw)))
                    .collect(),
            ),
        ),
        (
            "tensor_schema",
            Value::str(tensor_schema_digest_of(&plan_tensor_schema(&prepared.plan))),
        ),
        (
            "order",
            Value::arr(
                prepared
                    .plan
                    .ops
                    .iter()
                    .map(|operation| {
                        Value::arr(vec![
                            Value::str(operation.component.clone()),
                            Value::str(operation.out_key.clone()),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    object_id(&canon::write(&value))
        .trim_start_matches("sha256:")
        .chars()
        .take(16)
        .collect()
}

fn plan_source_prepared(
    registry_locator: &str,
    registry_bytes: &[u8],
    carriers: &[CarrierInput],
    requested_profile: Option<&str>,
    spec_variant: Option<usize>,
) -> Result<(SourcePlan, Prepared)> {
    if carriers.is_empty() {
        return refuse(Code::MISSING_FIELD, "source plan has no carriers");
    }
    // A MEMBER may not repeat; a PATH may. A Store is content addressed, so two members
    // carrying identical bytes resolve to one file: H3's FL2VA and Ref2VA transformer
    // indexes are byte-identical, so its 48 members present 47 paths. Assignment is by
    // member and `read_carrier` is a read, so a repeated path costs one extra header read
    // and decides nothing.
    let mut unique: Vec<CarrierInput> = Vec::with_capacity(carriers.len());
    for carrier in carriers {
        if !unique.contains(carrier) {
            unique.push(carrier.clone());
        }
    }
    let carriers = unique.as_slice();
    let mut members: Vec<&str> = carriers
        .iter()
        .filter_map(|carrier| carrier.member.as_deref())
        .collect();
    let member_count = members.len();
    members.sort();
    members.dedup();
    if members.len() != member_count {
        return refuse(
            Code::DUPLICATE_KEY,
            "one reviewed member names two different carriers",
        );
    }
    for carrier in carriers {
        if let Some(member) = &carrier.member {
            fingerprint::validate_source_member(member)?;
        }
    }
    let registry = FingerprintRegistry::parse(registry_bytes)?;
    if let Some(name) = requested_profile.filter(|name| *name != AS_IS) {
        if !registry
            .source_profiles
            .iter()
            .any(|profile| profile.name == name)
        {
            return refuse(
                Code::UNREGISTERED_FINGERPRINT,
                format!("reviewed source profile {name:?} is absent"),
            );
        }
    }
    // The selection is the directory. Built once, before a single header is read, so the
    // plan's header pass and `prepare`'s cannot disagree about which file a
    // `weight_map` value names.
    let set = CarrierSet::of(carriers);
    let mut headers = Vec::with_capacity(carriers.len());
    for carrier in carriers {
        headers.push(read_carrier(carrier, &set)?.0);
    }
    let mut matches = Vec::new();
    let mut requires_declaration = Vec::new();
    let mut rejected = None;
    for profile in &registry.source_profiles {
        if requested_profile.is_some_and(|name| name != profile.name) {
            continue;
        }
        let (solutions, refusal) = profile_assignments(profile, &registry, carriers, &headers);
        rejected = rejected.or(refusal);
        for assignments in solutions {
            if requested_profile.is_none() && !profile.auto_select {
                requires_declaration.push(profile.name.as_str());
                continue;
            }
            matches.push((profile, assignments));
        }
    }
    if !requires_declaration.is_empty() {
        return refuse(
            Code::AMBIGUOUS_CLASSIFICATION,
            format!(
                "headers cannot establish source layout; explicitly declare a reviewed format \
                 profile after checking the exporter contract: {requires_declaration:?}"
            ),
        );
    }
    if matches.is_empty() {
        if requested_profile.is_some_and(|name| name != AS_IS) {
            return Err(rejected.unwrap_or(Refusal {
                code: Code::UNREGISTERED_FINGERPRINT,
                detail: "carrier set does not match the requested reviewed source profile".into(),
            }));
        }
        let (profile, assignments) = as_is_profile(carriers, &headers)?;
        return plan_match(
            &profile,
            assignments,
            carriers,
            &set,
            None,
            registry_locator,
            registry_bytes,
            spec_variant,
        );
    }
    // Every match is planned. Matches whose plans are identical (the same sources, order,
    // target and session under different profile names) are one answer, named by the most
    // specific profile; only genuinely different plans are ambiguous.
    let mut planned = Vec::with_capacity(matches.len());
    let mut first_error = None;
    for (profile, assignments) in matches {
        match plan_match(
            profile,
            assignments,
            carriers,
            &set,
            Some(&registry),
            registry_locator,
            registry_bytes,
            spec_variant,
        ) {
            Ok(result) => planned.push((profile, result)),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    if planned.is_empty() {
        return Err(first_error.expect("every match was planned or refused"));
    }
    let same = |left: &SourcePlan, right: &SourcePlan| {
        (
            &left.session,
            &left.construction_order,
            &left.sources,
            &left.target,
        ) == (
            &right.session,
            &right.construction_order,
            &right.sources,
            &right.target,
        )
    };
    if planned
        .iter()
        .any(|(_, (plan, _))| !same(plan, &planned[0].1 .0))
    {
        let mut profiles: Vec<&str> = planned
            .iter()
            .map(|(profile, _)| profile.name.as_str())
            .collect();
        profiles.dedup();
        let mut members: Vec<String> = planned
            .iter()
            .flat_map(|(_, (plan, _))| {
                plan.sources.iter().map(|source| {
                    source
                        .source_member
                        .clone()
                        .unwrap_or_else(|| source.path.display().to_string())
                })
            })
            .collect();
        members.sort();
        members.dedup();
        return refuse(
            Code::AMBIGUOUS_CLASSIFICATION,
            format!(
                "multiple tensor selections match profiles {profiles:?}; select an exact file \
                 URL or source profile. Candidate members: {members:?}"
            ),
        );
    }
    planned.sort_by_key(|(profile, _)| {
        (
            std::cmp::Reverse(specificity(profile)),
            profile.name.clone(),
        )
    });
    let (_, (plan, prepared)) = planned.swap_remove(0);
    Ok((plan, prepared))
}

/// How exactly a profile binds its components: provider members, then member prefixes.
fn specificity(profile: &SourceProfile) -> (usize, usize) {
    let exact = profile
        .components
        .iter()
        .filter(|component| component.source_member.is_some())
        .count();
    let prefixed = profile
        .components
        .iter()
        .filter(|component| component.source_member_prefix.is_some())
        .count();
    (exact, prefixed)
}

/// The as-is recipe: one plain component per carrier a default narrowing keeps, in its own
/// key order. Shards come with their index. Carriers in folders win over loose root files
/// (alternative packagings); within a folder, or among root files of one stem, the
/// non-variant file wins, else the lowest precision (`x.fp16.safetensors` over
/// `x.fp32.safetensors`). A lone component is `model`; several are named by folder or stem.
fn as_is_profile(
    carriers: &[CarrierInput],
    headers: &[SourceHeader],
) -> Result<(SourceProfile, Vec<SourceAssignment>)> {
    let shards: std::collections::BTreeSet<&Path> = headers
        .iter()
        .flat_map(|header| header.shards.iter().map(|shard| shard.path.as_path()))
        .collect();
    let named: Vec<(usize, String)> = carriers
        .iter()
        .enumerate()
        .filter(|(_, carrier)| !shards.contains(carrier.path.as_path()))
        .map(|(index, carrier)| {
            let name = carrier.member.clone().unwrap_or_else(|| {
                let name = carrier.path.file_name().unwrap_or_default();
                name.to_string_lossy().into_owned()
            });
            (index, name)
        })
        .collect();
    let foldered = named.iter().any(|(_, name)| name.contains('/'));
    let mut ranked: Vec<_> = named
        .into_iter()
        .filter(|(_, name)| name.contains('/') == foldered)
        .map(|(index, name)| {
            let file = name.rsplit('/').next().unwrap_or(&name);
            let base = [".safetensors.index.json", ".safetensors", ".json"]
                .iter()
                .find_map(|suffix| file.strip_suffix(suffix))
                .unwrap_or(file);
            let (stem, variant) = base.split_once('.').unwrap_or((base, ""));
            let group = name.split_once('/').map_or(stem, |(folder, _)| folder);
            let precision = ["8", "16", "32", "64"]
                .iter()
                .position(|bits| variant.contains(bits));
            let rank = (
                !variant.is_empty(),
                precision.unwrap_or(4),
                variant.to_string(),
            );
            (group.to_string(), rank, name.to_string(), index)
        })
        .collect();
    // Sorted by group, then preference: the first of each group is its default.
    ranked.sort();
    ranked.dedup_by(|later, kept| later.0 == kept.0);
    let lone = ranked.len() == 1;
    let mut components: Vec<SourceProfileComponent> = Vec::new();
    let mut assignments = Vec::new();
    for (group, _, _, index) in ranked {
        let mut component: String = if lone {
            "model".into()
        } else {
            group
                .chars()
                .map(|c| match c {
                    'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.' => c,
                    _ => '_',
                })
                .collect()
        };
        while components.iter().any(|c| c.component == component) {
            component.push('_');
        }
        ascii_name("as-is component", &component, crate::limits::MAX_NAME_BYTES)?;
        components.push(SourceProfileComponent {
            component: component.clone(),
            source_member: carriers[index].member.clone(),
            source_member_prefix: None,
            target_encoding: "plain/1".into(),
            variants: vec![fingerprint::SourceProfileVariant {
                keyset_digest: String::new(),
                construction_order: headers[index]
                    .tensors
                    .iter()
                    .map(|t| t.key.clone())
                    .collect(),
            }],
        });
        assignments.push(SourceAssignment {
            component,
            carrier: index,
            projected: false,
            variant: 0,
        });
    }
    if components.is_empty() {
        return refuse(Code::MISSING_FIELD, "the source has no tensor carrier");
    }
    Ok((
        SourceProfile {
            auto_select: true,
            grammar: None,
            name: AS_IS.into(),
            components,
        },
        assignments,
    ))
}

#[allow(clippy::too_many_arguments)]
fn plan_match(
    profile: &SourceProfile,
    mut assignments: Vec<SourceAssignment>,
    carriers: &[CarrierInput],
    set: &CarrierSet,
    registry: Option<&FingerprintRegistry>,
    registry_locator: &str,
    registry_bytes: &[u8],
    spec_variant: Option<usize>,
) -> Result<(SourcePlan, Prepared)> {
    assignments.sort_by(|left, right| left.component.cmp(&right.component));
    let sources: Vec<(String, CarrierInput)> = assignments
        .iter()
        .map(|assignment| {
            (
                assignment.component.clone(),
                carriers[assignment.carrier].clone(),
            )
        })
        .collect();
    let target = profile
        .components
        .iter()
        .map(|component| format!("{}={}", component.component, component.target_encoding))
        .collect::<Vec<_>>()
        .join(",");
    let mut order: Vec<(String, String)> = profile
        .components
        .iter()
        .flat_map(|component| {
            let assignment = assignments
                .iter()
                .find(|assignment| assignment.component == component.component)
                .expect("profile assignment covers every component");
            let variant = component.variants.get(assignment.variant);
            variant
                .into_iter()
                .flat_map(|variant| &variant.construction_order)
                .map(|key| (component.component.clone(), key.clone()))
        })
        .collect();
    let classify = match registry {
        Some(registry) => classify(registry, &profile.name)?,
        None => Classify::AsIs,
    };
    let mut prepared = prepare(&target, &sources, set, classify, spec_variant)?;
    if profile.grammar.is_some() {
        // A grammar banks no traversal: its keys construct in sorted order.
        let ops = prepared.plan.ops.iter();
        order = ops
            .map(|op| (op.component.clone(), op.out_key.clone()))
            .collect();
        order.sort();
    }
    prepared.plan.apply_order(&order)?;
    prepared.plan.check_converts_something()?;
    let plan = SourcePlan {
        carriers: carriers.to_vec(),
        construction_order: prepared
            .plan
            .ops
            .iter()
            .map(|operation| (operation.component.clone(), operation.out_key.clone()))
            .collect(),
        converter: prepared.plan.converter.clone(),
        profile: profile.name.clone(),
        registry: registry_locator.to_string(),
        registry_sha256: format!("sha256:{}", crate::sha256::hex_digest(registry_bytes)),
        session: session_of(&prepared),
        sources: assignments
            .iter()
            .map(|assignment| PlannedSource {
                component: assignment.component.clone(),
                path: carriers[assignment.carrier].path.clone(),
                source_member: carriers[assignment.carrier].member.clone(),
                projected: assignment.projected,
            })
            .collect(),
        target,
    };
    Ok((plan, prepared))
}

pub fn plan_source(
    registry_locator: &str,
    registry_bytes: &[u8],
    carriers: &[CarrierInput],
    requested_profile: Option<&str>,
    spec_variant: Option<usize>,
) -> Result<SourcePlan> {
    plan_source_prepared(
        registry_locator,
        registry_bytes,
        carriers,
        requested_profile,
        spec_variant,
    )
    .map(|(plan, _)| plan)
}

pub fn golden_bytes(name: &str, length: u64) -> Vec<u8> {
    let mut seed = u64::from_le_bytes(
        crate::sha256::digest(name.as_bytes())[0..8]
            .try_into()
            .unwrap(),
    );
    let mut source = vec![0u8; length as usize];
    for chunk in source.chunks_mut(8) {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = seed;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let bytes = (value ^ (value >> 31)).to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    source
}

pub fn golden_suite(converter: &Converter) -> Result<usize> {
    let mut count = 0;
    for (name, transform) in convert::golden_cases(converter) {
        let source = golden_bytes(name, transform.in_bytes());
        if !transform.reassembles(&source)? {
            return refuse(
                Code::GOLDEN_MISMATCH,
                format!("{}: golden case {name} does not reassemble", converter.name),
            );
        }
        count += 1;
    }
    Ok(count)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSourceProfile {
    pub slot: String,
    pub profile: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModelSourceFile {
    pub member: String,
    pub object_id: String,
    pub length: u64,
    pub path: PathBuf,
    /// Exact index JSON or safetensors length-plus-header prefix, never payload.
    pub header: Vec<u8>,
}

/// The full immutable source roster and the bodies currently available. Arrival and local
/// paths do not identify the work. Output stays in the local Store until the RecordOwner
/// uploads and acknowledges its checkpoint chain; this call never waits for a cache mount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareModelSource {
    pub operation_id: String,
    pub source_selection_digest: String,
    pub profiles: Vec<ModelSourceProfile>,
    pub roster: Vec<VerifiedModelSourceFile>,
    pub landed: Vec<String>,
    /// Latest observed local heads. Only the caller knows which were remotely acknowledged.
    pub checkpoints: Vec<(String, ObjectRef)>,
    /// Explicit same-Store reuse. The supplied heads remain owned by this predecessor;
    /// preparation creates new operation-bound heads and independent retention roots.
    /// An empty checkpoint set snapshots the stopped predecessor's local journal first.
    pub adopt_from_operation_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelSource {
    pub slot: String,
    pub profile: String,
    pub manifest_digest: String,
    pub manifest_length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelSources {
    pub replayed: bool,
    pub complete: bool,
    pub sources: Vec<PreparedModelSource>,
    pub checkpoints: Vec<SourceCheckpoint>,
    pub spent: Vec<String>,
    pub converted_roles: usize,
    pub resumed_roles: usize,
    pub deferred_ops: usize,
    pub converted_bytes: u64,
    pub largest_op_bytes: u64,
    /// Minimum body admission for the next atomic op, without writing that op.
    pub required_write_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceCheckpoint {
    pub slot: String,
    pub plan_digest: String,
    /// Locally checkpointed bytes. Remote durability requires the caller's upload acknowledgment.
    pub head: durability::Head,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileFence {
    path: PathBuf,
    dev: u64,
    ino: u64,
    length: u64,
    mtime_s: i64,
    mtime_ns: i64,
}

impl FileFence {
    fn capture(file: &VerifiedModelSourceFile) -> Result<Self> {
        if !file.path.is_absolute() {
            return refuse(Code::KEY_GRAMMAR, "verified source path is not absolute");
        }
        fingerprint::validate_source_member(&file.member)?;
        // pod-supervisor already streamed and SHA-verified this exact immutable file. A
        // second whole-file pass here would double 200+ GiB of I/O before ingest reads the
        // same bytes again. TensorFS binds that verified ObjectRef into operation identity,
        // re-fences the inode/change token, and hashes every admitted tensor byte in the
        // ordinary segmentation pass below.
        prefixed("verified source ObjectRef", &file.object_id)?;
        if file.length == 0 {
            return refuse(
                Code::LENGTH_MISMATCH,
                "verified source file has zero length",
            );
        }
        let metadata = fs::symlink_metadata(&file.path).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("stat {}: {error}", file.path.display()),
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return refuse(
                Code::KEY_GRAMMAR,
                "verified source path is not a regular non-symlink file",
            );
        }
        if metadata.len() != file.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{} is {} bytes, expected {}",
                    file.member,
                    metadata.len(),
                    file.length
                ),
            );
        }
        Ok(Self {
            path: file.path.clone(),
            dev: metadata.dev(),
            ino: metadata.ino(),
            length: metadata.len(),
            mtime_s: metadata.mtime(),
            mtime_ns: metadata.mtime_nsec(),
        })
    }

    fn check(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.path).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("restat {}: {error}", self.path.display()),
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.dev() != self.dev
            || metadata.ino() != self.ino
            || metadata.len() != self.length
            || metadata.mtime() != self.mtime_s
            || metadata.mtime_nsec() != self.mtime_ns
        {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "verified source file changed during preparation",
            );
        }
        Ok(())
    }
}

/// Sort by `key` and merge identical rows. Two different rows under one key still refuse.
fn merge_rows<T: Clone + PartialEq, K: Ord>(
    what: &str,
    rows: &mut Vec<T>,
    key: impl Fn(&T) -> K,
) -> Result<()> {
    rows.sort_by_key(&key);
    rows.dedup();
    if rows.windows(2).any(|pair| key(&pair[0]) == key(&pair[1])) {
        return refuse(
            Code::DUPLICATE_KEY,
            format!("{what} names one key twice with different values"),
        );
    }
    Ok(())
}

/// The caller's rows in the one order the digest, plan and replay use.
fn normalized(request: &PrepareModelSource) -> Result<PrepareModelSource> {
    let mut request = request.clone();
    merge_rows("source profiles", &mut request.profiles, |row| {
        row.slot.clone()
    })?;
    merge_rows("verified source files", &mut request.roster, |row| {
        row.member.clone()
    })?;
    merge_rows("source checkpoints", &mut request.checkpoints, |row| {
        row.0.clone()
    })?;
    request.landed.sort();
    request.landed.dedup();
    Ok(request)
}

fn validate_request(
    request: &PrepareModelSource,
    registry: &FingerprintRegistry,
) -> Result<(std::collections::BTreeSet<PathBuf>, Vec<FileFence>)> {
    crate::catalog::source_operation_id(&request.operation_id)?;
    if let Some(prior) = &request.adopt_from_operation_id {
        crate::catalog::source_operation_id(prior)?;
        if prior == &request.operation_id || request.checkpoints.is_empty() {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "source adoption requires another operation and its checkpoint heads",
            );
        }
    }
    prefixed("source selection digest", &request.source_selection_digest)?;
    if request.profiles.is_empty() || request.profiles.len() > 16 {
        return refuse(Code::COUNT_CAP, "source profile count is outside 1..=16");
    }
    for row in &request.profiles {
        ascii_name("source slot", &row.slot, crate::limits::MAX_NAME_BYTES)?;
        if row.profile != AS_IS
            && !registry
                .source_profiles
                .iter()
                .any(|profile| profile.name == row.profile)
        {
            return refuse(
                Code::UNREGISTERED_FINGERPRINT,
                format!("source profile {:?} is not reviewed", row.profile),
            );
        }
    }
    if request.roster.is_empty() || request.roster.len() > 4096 {
        return refuse(
            Code::COUNT_CAP,
            "verified source file count is outside 1..=4096",
        );
    }
    // Same rule as the plan above, stated where the object ids are still in hand: a path
    // is an OBJECT's identity, so it may repeat across members, but only ever bearing one
    // object. Two DIFFERENT objects on one path is a real fault and stays a refusal.
    let mut owners: BTreeMap<&Path, &str> = BTreeMap::new();
    for row in &request.roster {
        if *owners
            .entry(row.path.as_path())
            .or_insert(row.object_id.as_str())
            != row.object_id
        {
            return refuse(
                Code::DUPLICATE_KEY,
                format!(
                    "verified source path {} carries two different objects",
                    row.path.display()
                ),
            );
        }
    }
    let mut ready = std::collections::BTreeSet::new();
    let mut fences = Vec::new();
    for row in &request.roster {
        fingerprint::validate_source_member(&row.member)?;
        prefixed("source ObjectRef", &row.object_id)?;
        if !row.path.is_absolute()
            || row.length == 0
            || row.header.is_empty()
            || row.header.len() > crate::limits::CARRIER_HEADER_MAX_BYTES + 8
            || row.header.len() as u64 > row.length
        {
            return refuse(
                Code::CARRIER_GEOMETRY,
                "source roster has invalid path, length or bounded header",
            );
        }
        if row.member.ends_with(".index.json") {
            if row.header.len() as u64 != row.length
                || ObjectRef::of(&row.header).id() != row.object_id
            {
                return refuse(
                    Code::OBJECT_ID_MISMATCH,
                    "index metadata differs from its pinned ObjectRef",
                );
            }
        } else {
            if row.header.len() < 8 {
                return refuse(
                    Code::CARRIER_TRUNCATED,
                    "source header has no length prefix",
                );
            }
            let n = u64::from_le_bytes(row.header[..8].try_into().expect("eight bytes"));
            if n != row.header.len() as u64 - 8 {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    "source header length differs from its exact prefix",
                );
            }
            carrier::parse_header(&row.header[8..], n + 8, Some(row.length))?;
        }
    }
    for member in &request.landed {
        let row = request
            .roster
            .iter()
            .find(|row| &row.member == member)
            .ok_or_else(|| Refusal {
                code: Code::MISSING_FIELD,
                detail: "landed source is outside the pinned roster".into(),
            })?;
        fences.push(FileFence::capture(row)?);
        // The header used to plan must be the prefix of the admitted body, not a second
        // unauthenticated description of it. This reads only metadata, never the payload.
        let mut file = fs::File::open(&row.path).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("open source header: {error}"),
        })?;
        let mut prefix = vec![0; row.header.len()];
        file.read_exact(&mut prefix).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("read source header: {error}"),
        })?;
        if prefix != row.header {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "landed carrier header changed from its pinned metadata",
            );
        }
        ready.insert(row.path.clone());
    }
    let mut checkpoint_slots = std::collections::BTreeSet::new();
    for (slot, _) in &request.checkpoints {
        if !checkpoint_slots.insert(slot) || !request.profiles.iter().any(|row| &row.slot == slot) {
            return refuse(
                Code::DUPLICATE_KEY,
                "checkpoint slot repeats or is outside the profile set",
            );
        }
    }
    Ok((ready, fences))
}

fn digest_field(hash: &mut crate::sha256::Sha256, value: &[u8]) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn request_digest(request: &PrepareModelSource, _registry_bytes: &[u8]) -> String {
    let mut hash = crate::sha256::Sha256::new();
    digest_field(&mut hash, b"tensorfs.model-source-preparation/2");
    digest_field(&mut hash, request.source_selection_digest.as_bytes());
    for row in &request.profiles {
        digest_field(&mut hash, row.slot.as_bytes());
        digest_field(&mut hash, row.profile.as_bytes());
    }
    for row in &request.roster {
        digest_field(&mut hash, row.member.as_bytes());
        digest_field(&mut hash, row.object_id.as_bytes());
        digest_field(&mut hash, &row.length.to_le_bytes());
        digest_field(&mut hash, &row.header);
    }
    format!("sha256:{}", crate::sha256::hex(&hash.finish()))
}

fn validate_prepared_result(
    store: &Store,
    source: &PreparedModelSource,
    custody: &super::custody::Custody,
) -> Result<()> {
    let manifest_ref = ObjectRef {
        sha256: source
            .manifest_digest
            .strip_prefix("sha256:")
            .unwrap_or(&source.manifest_digest)
            .to_string(),
        length: source.manifest_length,
    };
    let manifest = store.read_manifest(&manifest_ref)?;
    let header_ref = manifest.header().ok_or_else(|| Refusal {
        code: Code::MISSING_FIELD,
        detail: "prepared model Manifest has no CozyTensors header".into(),
    })?;
    crate::checkpoint::load_header(store, header_ref)?;
    crate::checkpoint::walk_cozytensors(store, &manifest)?
        .require_held(store, &custody.digests())?;
    Ok(())
}

pub fn prepare_model_source_with_registry(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
) -> Result<PreparedModelSources> {
    prepare_model_source_with_registry_budget(
        store,
        request,
        registry_locator,
        registry_bytes,
        None,
    )
}

/// An explicit body budget never admits an oversized op. Zero advances only metadata
/// and replay, reporting the next atomic body's native byte count for host admission.
pub fn prepare_model_source_with_registry_budget(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
    write_budget_bytes: Option<u64>,
) -> Result<PreparedModelSources> {
    let request = &normalized(request)?;
    // The guard bridges local snapshot publication and recipient root adoption. A
    // predecessor can be released between them, but GC cannot remove the saved bytes.
    let _guard = crate::catalog::WriterGuard::acquire(store.root())?;
    let mut resumed;
    let request = if let Some(prior) = request
        .adopt_from_operation_id
        .as_ref()
        .filter(|_| request.checkpoints.is_empty())
    {
        if prior == &request.operation_id {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "source adoption requires another operation",
            );
        }
        let operations = crate::catalog::Catalog::open(store.root())?.model_source_operations()?;
        let own_work = operations.contains(&request.operation_id);
        let mut original = request.clone();
        original.adopt_from_operation_id = None;
        let own_heads = if own_work {
            snapshot_model_source(store, &original, registry_locator, registry_bytes)?
        } else {
            Vec::new()
        };
        resumed = request.clone();
        let heads = if own_heads.len() < request.profiles.len() && operations.contains(prior) {
            original.operation_id = prior.clone();
            let heads = snapshot_model_source(store, &original, registry_locator, registry_bytes)?;
            if heads.is_empty() {
                resumed.adopt_from_operation_id = None;
                own_heads
            } else {
                // The normal per-profile adoption path prefers each recipient's local
                // head, then uses the predecessor only for profiles not acquired yet.
                heads
            }
        } else if own_work {
            resumed.adopt_from_operation_id = None;
            own_heads
        } else {
            return refuse(
                Code::ROOT_ABSENT,
                "source adoption has no retained predecessor",
            );
        };
        resumed.checkpoints = heads
            .into_iter()
            .filter_map(|checkpoint| checkpoint.head.head.map(|head| (checkpoint.slot, head)))
            .collect();
        &resumed
    } else {
        request
    };
    prepare_model_source_with_budget(
        store,
        request,
        registry_locator,
        registry_bytes,
        &mut match write_budget_bytes {
            Some(limit) => transaction::Budget::bounded(limit),
            None => transaction::Budget::new(durability::INTERVAL_BYTES),
        },
        true,
    )
}

struct PlannedPreparation {
    ready: std::collections::BTreeSet<PathBuf>,
    fences: Vec<FileFence>,
    plans: Vec<(String, Prepared)>,
    digest: String,
}

fn plan_preparation(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
) -> Result<PlannedPreparation> {
    let registry = FingerprintRegistry::parse(registry_bytes)?;
    let (ready, fences) = validate_request(request, &registry)?;
    let digest = request_digest(request, registry_bytes);
    let heads: Vec<_> = request
        .roster
        .iter()
        .map(|row| crate::providers::MemberHead {
            member: row.member.clone(),
            length: row.length,
            head: row.header.clone(),
        })
        .collect();
    let staged = super::preflight::stage(
        &store
            .root()
            .join("tmp")
            .join(format!("source-heads-{}", crate::meta::now_nanos_unique())),
        &heads,
    )?;
    let locations: BTreeMap<&Path, &VerifiedModelSourceFile> = staged
        .carriers()
        .iter()
        .zip(&request.roster)
        .map(|(carrier, row)| (carrier.path.as_path(), row))
        .collect();
    let mut plans = Vec::new();
    for requested in &request.profiles {
        let (_, mut prepared) = plan_source_prepared(
            registry_locator,
            registry_bytes,
            staged.carriers(),
            Some(&requested.profile),
            None,
        )?;
        let _ = golden_suite(prepared.conv)?;
        for (_, seed) in &prepared.seeds {
            if let Some(vectors) = &seed.vectors {
                vectors.check(&seed.spec)?;
            }
        }
        // The tensor plan binds transforms/geometry/order; the pinned request binds all
        // carrier object identities. Locations participate in neither identity.
        let stable_headers: Vec<_> = prepared
            .files
            .iter()
            .map(|file| {
                let row = locations[&file.path.as_path()];
                (PathBuf::from(&row.member), &file.header)
            })
            .collect();
        let tensor_plan = super::journal::plan_digest(&prepared.plan, &stable_headers);
        let mut hash = crate::sha256::Sha256::new();
        digest_field(&mut hash, digest.as_bytes());
        digest_field(&mut hash, requested.slot.as_bytes());
        digest_field(&mut hash, tensor_plan.as_bytes());
        let plan_hex = crate::sha256::hex(&hash.finish());
        for file in &mut prepared.files {
            file.path = locations[&file.path.as_path()].path.clone();
            for shard in &mut file.header.shards {
                shard.path = locations[&shard.path.as_path()].path.clone();
            }
        }
        plans.push((plan_hex, prepared));
    }
    // A completed result replays only after its actual tensor plans were re-derived.
    // Unrelated registry edits do not change these plans or invalidate completed work.
    let mut request_hash = crate::sha256::Sha256::new();
    digest_field(&mut request_hash, digest.as_bytes());
    for (plan, _) in &plans {
        digest_field(&mut request_hash, plan.as_bytes());
    }
    let digest = format!("sha256:{}", crate::sha256::hex(&request_hash.finish()));
    Ok(PlannedPreparation {
        ready,
        fences,
        plans,
        digest,
    })
}

/// `chain` writes the durability chain a RecordOwner uploads for cross-pod resume. A
/// custodied operation's durable progress is its publication instead, so it has no chain.
fn prepare_model_source_with_budget(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
    budget: &mut transaction::Budget,
    chain: bool,
) -> Result<PreparedModelSources> {
    let PlannedPreparation {
        ready,
        fences,
        plans,
        digest,
    } = plan_preparation(store, request, registry_locator, registry_bytes)?;
    let custody = super::custody::read(store.root(), &request.operation_id)?;
    if chain && !custody.is_empty() {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "a custodied source operation has no local durability chain",
        );
    }
    let catalog = crate::catalog::Catalog::open(store.root())?;
    let writer = match catalog.begin_source_preparation(&request.operation_id, &digest)? {
        crate::catalog::SourcePreparation::Prepared(rows) => {
            if rows.len() != request.profiles.len()
                || rows
                    .iter()
                    .zip(&request.profiles)
                    .any(|(stored, requested)| {
                        stored.slot != requested.slot || stored.profile != requested.profile
                    })
            {
                return refuse(
                    Code::DURABILITY_UNPROVEN,
                    "stored source results differ from their exact request",
                );
            }
            let sources: Vec<_> = rows
                .into_iter()
                .map(|row| PreparedModelSource {
                    slot: row.slot,
                    profile: row.profile,
                    manifest_digest: format!("sha256:{}", row.manifest_sha256),
                    manifest_length: row.manifest_length,
                })
                .collect();
            for source in &sources {
                validate_prepared_result(store, source, &custody)?;
            }
            // The final response can be lost after its catalog commit. Its existing local
            // cursors must remain discoverable without converting or reading source bodies.
            let mut checkpoints = Vec::new();
            let dir = transaction::candidate_dir(store.root(), &request.operation_id);
            for (requested, (plan_hex, _)) in request.profiles.iter().zip(&plans).filter(|_| chain)
            {
                let plan_digest = format!("sha256:{plan_hex}");
                let chain = format!("{}/{}", request.operation_id, requested.slot);
                let journal = super::journal::Journal::open(&dir, plan_hex)?;
                let local = journal.local_head()?.ok_or_else(|| Refusal {
                    code: Code::DURABILITY_UNPROVEN,
                    detail: "completed source has no local checkpoint cursor".into(),
                })?;
                let mut supplied = request
                    .checkpoints
                    .iter()
                    .filter(|_| request.adopt_from_operation_id.is_none())
                    .find(|(slot, _)| slot == &requested.slot)
                    .map(|(_, head)| head);
                if let Some(head) = supplied {
                    if durability::continuable(store, head, &plan_digest, &chain)?.is_none() {
                        supplied = None;
                    }
                }
                let head = durability::checkpoint_predecessor(
                    store,
                    &durability::Policy {
                        chain: &chain,
                        operation: &request.operation_id,
                        plan: &plan_digest,
                        interval: durability::INTERVAL_BYTES,
                    },
                    Some(&local),
                    supplied,
                )?
                .expect("a local cursor was supplied");
                let held = durability::local_chain(store, &head, &plan_digest)?;
                checkpoints.push(SourceCheckpoint {
                    slot: requested.slot.clone(),
                    plan_digest,
                    head: durability::Head {
                        head: Some(head),
                        links: held.links,
                        bytes: held.bytes,
                    },
                });
            }
            return Ok(PreparedModelSources {
                replayed: true,
                complete: true,
                sources,
                checkpoints,
                spent: request
                    .roster
                    .iter()
                    .map(|row| row.member.clone())
                    .collect(),
                converted_roles: 0,
                resumed_roles: 0,
                deferred_ops: 0,
                converted_bytes: 0,
                largest_op_bytes: 0,
                required_write_bytes: 0,
            });
        }
        crate::catalog::SourcePreparation::Open(writer) => writer,
    };
    transaction::ensure_session_root(store.root(), &request.operation_id, "_tensorfs")?;
    let mut result = PreparedModelSources {
        replayed: false,
        complete: true,
        sources: Vec::new(),
        checkpoints: Vec::new(),
        spent: Vec::new(),
        converted_roles: 0,
        resumed_roles: 0,
        deferred_ops: 0,
        converted_bytes: 0,
        largest_op_bytes: 0,
        required_write_bytes: 0,
    };
    let mut needed = std::collections::BTreeSet::new();
    if !chain {
        // Convert every slot first; finalize below only once all of them are journalled.
        let dir = transaction::candidate_dir(store.root(), &request.operation_id);
        for (plan_hex, prepared) in &plans {
            let mut journal = super::journal::Journal::open(&dir, plan_hex)?;
            for fence in &fences {
                fence.check()?;
            }
            let progress = transaction::convert_journal(
                store,
                &prepared.plan,
                &prepared.files,
                &request.operation_id,
                transaction::Carriers::Ready(&ready),
                &mut journal,
                Some(budget),
                &custody,
            )?;
            needed.extend(progress.needed);
            result.converted_roles += progress.converted_roles;
            result.converted_bytes += progress.written;
            result.resumed_roles += progress.resumed_roles;
            result.deferred_ops += progress.deferred_ops;
        }
        result.largest_op_bytes = budget.largest_op;
        result.required_write_bytes = budget.required_write_bytes;
        if result.deferred_ops > 0 {
            for fence in &fences {
                fence.check()?;
            }
            result.complete = false;
            result.spent = request
                .roster
                .iter()
                .filter(|row| !needed.contains(&row.path))
                .map(|row| row.member.clone())
                .collect();
            drop(writer);
            return Ok(result);
        }
    }
    for (requested, (plan_hex, prepared)) in request.profiles.iter().zip(plans) {
        let plan_digest = format!("sha256:{plan_hex}");
        let chain_name = format!("{}/{}", request.operation_id, requested.slot);
        let mut previous = request
            .checkpoints
            .iter()
            .find(|(slot, _)| slot == &requested.slot)
            .map(|(_, head)| head.clone());
        let dir = transaction::candidate_dir(store.root(), &request.operation_id);
        let mut journal = super::journal::Journal::open(&dir, &plan_hex)?;
        if let Some(prior) = &request.adopt_from_operation_id {
            previous = if let Some(local) = journal.local_head()? {
                // A repeated adoption call must continue its own newer progress even
                // after the original operation has released its independent root.
                Some(local)
            } else if let Some(head) = &previous {
                adopt_source_checkpoint(
                    store,
                    prior,
                    &chain_name,
                    &request.operation_id,
                    &plan_hex,
                    head,
                    &dir,
                    &mut journal,
                )?
                .head
            } else {
                None
            };
        }
        let held = match &previous {
            Some(head) => durability::continuable(store, head, &plan_digest, &chain_name)?,
            None => None,
        };
        if held.is_none() {
            previous = None;
        }
        discard_other_plans(store, &dir, &chain_name, &plan_hex)?;
        if let (Some(head), Some(held)) = (&previous, held) {
            durability::retain_local_chain(store, head, &request.operation_id)?;
            if journal.is_empty() {
                if let Some(progress) = held.progress.last() {
                    let bytes =
                        durability::read_document(store, progress, crate::limits::DOC_MAX_BYTES)?;
                    drop(journal);
                    super::journal::install(&dir, &plan_hex, &bytes)?;
                    journal = super::journal::Journal::open(&dir, &plan_hex)?;
                }
            }
        }
        for fence in &fences {
            fence.check()?;
        }
        let (progress, outcome) = transaction::advance_journal(
            store,
            &prepared.plan,
            &prepared.files,
            &plain_specs(&prepared.seeds),
            &[],
            &request.operation_id,
            transaction::Carriers::Ready(&ready),
            None,
            &mut journal,
            Some(budget),
            &custody,
        )?;
        if chain {
            let checkpoint = durability::checkpoint(
                store,
                &durability::Policy {
                    chain: &chain_name,
                    operation: &request.operation_id,
                    plan: &plan_digest,
                    interval: durability::INTERVAL_BYTES,
                },
                previous.as_ref(),
                &journal,
            )?;
            if checkpoint.head.is_some() {
                result.checkpoints.push(SourceCheckpoint {
                    slot: requested.slot.clone(),
                    plan_digest,
                    head: checkpoint,
                });
            }
        }
        needed.extend(progress.needed);
        if chain {
            result.converted_roles += progress.converted_roles;
            result.converted_bytes += progress.written;
            result.largest_op_bytes = budget.largest_op;
            result.required_write_bytes = budget.required_write_bytes;
            result.resumed_roles += progress.resumed_roles;
            result.deferred_ops += progress.deferred_ops;
        }
        if let Some(outcome) = outcome {
            if outcome.header.tensor_schema_digest()
                != tensor_schema_digest_of(&plan_tensor_schema(&prepared.plan))
            {
                return refuse(
                    Code::TENSOR_SCHEMA_MISMATCH,
                    "source plan and executed tensor schema differ",
                );
            }
            result.sources.push(PreparedModelSource {
                slot: requested.slot.clone(),
                profile: requested.profile.clone(),
                manifest_digest: outcome.manifest_ref.id(),
                manifest_length: outcome.manifest_ref.length,
            });
        } else {
            result.complete = false;
        }
    }
    for fence in &fences {
        fence.check()?;
    }
    result.spent = request
        .roster
        .iter()
        .filter(|row| !needed.contains(&row.path))
        .map(|row| row.member.clone())
        .collect();
    if result.complete {
        for source in &result.sources {
            validate_prepared_result(store, source, &custody)?;
        }
        let rows: Vec<_> = result
            .sources
            .iter()
            .map(|source| crate::catalog::SourcePreparationResult {
                slot: source.slot.clone(),
                profile: source.profile.clone(),
                manifest_sha256: source
                    .manifest_digest
                    .trim_start_matches("sha256:")
                    .to_string(),
                manifest_length: source.manifest_length,
            })
            .collect();
        catalog.commit_source_preparation(&request.operation_id, &digest, &rows)?;
    } else {
        result.sources.clear();
    }
    drop(writer);
    Ok(result)
}

pub fn prepare_model_source(
    store: &Store,
    request: &PrepareModelSource,
) -> Result<PreparedModelSources> {
    let (locator, bytes) = effective_registry(store, None)?;
    prepare_model_source_with_registry(store, request, &locator, &bytes)
}

/// One member of an exact provider selection: its pinned identity and the header a ranged
/// read bought before any body moved. Only index and tensor carriers belong here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedSourceMember {
    pub member: String,
    pub object: ObjectRef,
    /// Exact index JSON or safetensors length-plus-header prefix, never payload.
    pub header: Vec<u8>,
}

/// Advance an exact selection whose bodies arrive and leave while it converts, within a
/// bounded disk. A member is landed when the Store holds its verified body right now;
/// output the operation's custodian holds stands without local bytes. There is no
/// durability chain: the custodian is this operation's durable progress.
#[allow(clippy::too_many_arguments)]
pub fn prepare_selected_source(
    store: &Store,
    operation_id: &str,
    source_selection_digest: &str,
    profiles: Vec<ModelSourceProfile>,
    members: &[SelectedSourceMember],
    registry_locator: &str,
    registry_bytes: &[u8],
    write_budget_bytes: Option<u64>,
) -> Result<(PreparedModelSources, Vec<String>)> {
    // Presence is read under the writer fence, so GC cannot remove a body counted as landed.
    let _guard = crate::catalog::WriterGuard::acquire(store.root())?;
    let mut landed = Vec::new();
    let roster = members
        .iter()
        .map(|row| {
            if matches!(store.record_valid(&row.object.sha256), Ok(record) if record.length == row.object.length)
            {
                landed.push(row.member.clone());
            }
            VerifiedModelSourceFile {
                member: row.member.clone(),
                object_id: row.object.id(),
                length: row.object.length,
                path: store.object_path(&row.object.sha256),
                header: row.header.clone(),
            }
        })
        .collect();
    let request = normalized(&PrepareModelSource {
        operation_id: operation_id.to_string(),
        source_selection_digest: source_selection_digest.to_string(),
        profiles,
        roster,
        landed,
        checkpoints: Vec::new(),
        adopt_from_operation_id: None,
    })?;
    let prepared = prepare_model_source_with_budget(
        store,
        &request,
        registry_locator,
        registry_bytes,
        &mut match write_budget_bytes {
            Some(limit) => transaction::Budget::bounded(limit),
            None => transaction::Budget::new(u64::MAX),
        },
        false,
    )?;
    Ok((prepared, request.landed))
}

/// Reuse verified journal parts without relabeling the old chain or its ownership.
/// The destination writer fence is held by preparation. A temporary read hold spans
/// validation through installation of the new persistent root, including a concurrent
/// predecessor release. No tensor payload is copied and no source carrier is read.
#[allow(clippy::too_many_arguments)]
fn adopt_source_checkpoint(
    store: &Store,
    prior: &str,
    chain: &str,
    operation: &str,
    plan_hex: &str,
    head: &ObjectRef,
    dir: &Path,
    journal: &mut super::journal::Journal,
) -> Result<durability::Head> {
    let meta = crate::meta::Meta::open(store)?;
    let hold = meta.acquire_hold("source-adoption")?;
    let result = (|| {
        // Hold the predecessor's actual operation through independent recipient
        // admission. Its release either wins first (refusal) or observes this live
        // claim; resident unowned checkpoint bytes alone are never adoption authority.
        let catalog = crate::catalog::Catalog::open(store.root())?;
        let _predecessor =
            catalog.resume_retained_operation(prior, "_tensorfs", "model-source-preparation")?;
        let retained = transaction::read_session(store.root(), prior)?;
        if !retained.candidates.contains(head) {
            return refuse(
                Code::ROOT_ABSENT,
                "source checkpoint is outside its predecessor retention root",
            );
        }

        let plan = format!("sha256:{plan_hex}");
        let slot = chain
            .strip_prefix(&format!("{operation}/"))
            .expect("source slot chain");
        let Some(held) = durability::continuable(store, head, &plan, &format!("{prior}/{slot}"))?
        else {
            return Ok(durability::Head::default());
        };
        let installed = journal.is_empty();
        if installed {
            let progress = held.progress.last().ok_or_else(|| Refusal {
                code: Code::DURABILITY_UNPROVEN,
                detail: "source checkpoint has no conversion progress".into(),
            })?;
            let bytes = durability::read_document(store, progress, crate::limits::DOC_MAX_BYTES)?;
            super::journal::install(dir, plan_hex, &bytes)?;
            *journal = super::journal::Journal::open(dir, plan_hex)?;
        }
        // GC already discovers every part through this operation's installed journal.
        // Keep only checkpoint documents in the small session root; duplicating the
        // complete tensor roster there exceeds its independent metadata bound.
        // checkpoint verifies every part before returning the new head. A part that did not
        // survive leaves no head to adopt; the journal's own law reconverts it.
        match durability::checkpoint(
            store,
            &durability::Policy {
                chain,
                operation,
                plan: &plan,
                interval: durability::INTERVAL_BYTES,
            },
            None,
            journal,
        ) {
            Err(error) if error.code != Code::IO_FAILED => {
                // Nothing of a chain that did not survive is kept, not even its journal.
                if installed {
                    super::journal::discard(dir, plan_hex)?;
                    *journal = super::journal::Journal::open(dir, plan_hex)?;
                }
                Ok(durability::Head::default())
            }
            result => result,
        }
    })();
    let released = hold.release(&meta);
    result.and_then(|head| released.map(|()| head))
}

/// Drop this slot's journals and cursors written under any other plan. They can never be
/// continued, and keeping them would hold their bytes and leave two cursors for one slot.
fn discard_other_plans(store: &Store, dir: &Path, chain: &str, plan_hex: &str) -> Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(other) = name
            .strip_prefix("convert-")
            .and_then(|rest| rest.strip_suffix(".head.json"))
        else {
            continue;
        };
        if other == plan_hex {
            continue;
        }
        let head = fs::read(entry.path())
            .ok()
            .and_then(|bytes| canon::parse(&bytes, crate::limits::SPEC_MAX_BYTES).ok())
            .and_then(|value| ObjectRef::from_value("source progress head", &value).ok());
        let slot_of_other = head
            .and_then(|head| durability::local_link(store, &head).ok())
            .map(|link| link.operation);
        if slot_of_other.as_deref() != Some(chain) {
            continue;
        }
        super::journal::discard(dir, other)?;
    }
    Ok(())
}

pub fn release_model_source(store: &Store, operation_id: &str) -> Result<()> {
    crate::catalog::Catalog::open(store.root())?.release_source_preparation(operation_id, || {
        match fs::remove_dir_all(transaction::candidate_dir(store.root(), operation_id)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Refusal {
                code: Code::IO_FAILED,
                detail: format!("remove model-source journal: {error}"),
            }),
        }
    })
}

/// Read already-durable source heads without reopening a converter or rewriting a journal.
/// Each cursor is validated against its native chain and exact operation/plan identity.
pub fn source_checkpoints(store: &Store, operation: &str) -> Result<Vec<SourceCheckpoint>> {
    crate::catalog::source_operation_id(operation)?;
    let _writer = crate::catalog::WriterGuard::acquire(store.root())?;
    if !crate::catalog::Catalog::open(store.root())?
        .model_source_operations()?
        .iter()
        .any(|id| id == operation)
    {
        return refuse(Code::ROOT_ABSENT, "source operation is not retained");
    }
    let dir = transaction::candidate_dir(store.root(), operation);
    let mut checkpoints: BTreeMap<String, (std::time::SystemTime, SourceCheckpoint)> =
        BTreeMap::new();
    for entry in fs::read_dir(&dir).map_err(|e| Refusal {
        code: Code::IO_FAILED,
        detail: format!("source progress directory: {e}"),
    })? {
        let path = entry
            .map_err(|e| Refusal {
                code: Code::IO_FAILED,
                detail: format!("source progress entry: {e}"),
            })?
            .path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(plan_hex) = name
            .strip_prefix("convert-")
            .and_then(|value| value.strip_suffix(".head.json"))
        else {
            continue;
        };
        crate::ids::hex64("source progress plan", plan_hex)?;
        let mut bytes = Vec::new();
        fs::File::open(&path)
            .map_err(|e| Refusal {
                code: Code::IO_FAILED,
                detail: format!("source progress cursor: {e}"),
            })?
            .take(crate::limits::SPEC_MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| Refusal {
                code: Code::IO_FAILED,
                detail: format!("source progress read: {e}"),
            })?;
        let value = canon::parse_canonical(&bytes, crate::limits::SPEC_MAX_BYTES)?;
        let head = ObjectRef::from_value("source progress head", &value)?;
        let plan_digest = format!("sha256:{plan_hex}");
        let chain = durability::local_chain(store, &head, &plan_digest)?;
        let Some(slot) = chain.operation.strip_prefix(&format!("{operation}/")) else {
            return refuse(
                Code::CROSS_SUBJECT_REPLAY,
                "source cursor belongs to another operation",
            );
        };
        ascii_name("source progress slot", slot, crate::limits::MAX_NAME_BYTES)?;
        // A slot re-planned since (a changed converter or registry) keeps its newest cursor;
        // the next preparation discards the others.
        let written = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        if checkpoints
            .get(slot)
            .is_some_and(|(newer, _)| *newer >= written)
        {
            continue;
        }
        checkpoints.insert(
            slot.to_string(),
            (
                written,
                SourceCheckpoint {
                    slot: slot.to_string(),
                    plan_digest,
                    head: durability::Head {
                        head: Some(head),
                        links: chain.links,
                        bytes: chain.bytes,
                    },
                },
            ),
        );
    }
    Ok(checkpoints
        .into_values()
        .map(|(_, checkpoint)| checkpoint)
        .collect())
}

/// Export only already-fsynced conversion groups from a stopped exact source writer.
/// No missing conversion operation runs. Existing local checkpoint export supplies the
/// immutable head which a fresh request independently adopts through the normal engine.
pub fn snapshot_model_source(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
) -> Result<Vec<SourceCheckpoint>> {
    let request = &normalized(request)?;
    if request.adopt_from_operation_id.is_some() || !request.checkpoints.is_empty() {
        return refuse(
            Code::TRANSACTION_CONFLICT,
            "source snapshot names its own original declaration only",
        );
    }
    let catalog = crate::catalog::Catalog::open(store.root())?;
    if !catalog
        .model_source_operations()?
        .iter()
        .any(|id| id == &request.operation_id)
    {
        return refuse(
            Code::ROOT_ABSENT,
            "source snapshot requires a retained operation",
        );
    }
    transaction::read_session(store.root(), &request.operation_id)?;
    let PlannedPreparation {
        plans,
        digest,
        fences,
        ..
    } = plan_preparation(store, request, registry_locator, registry_bytes)?;
    // This existing operation claim refuses a live predecessor. Its persisted request
    // digest also refuses edited converter code, source identity, or selected plans.
    let _writer = match catalog.begin_retained_source_preparation(&request.operation_id, &digest)? {
        crate::catalog::SourcePreparation::Prepared(_) => {
            return source_checkpoints(store, &request.operation_id)
        }
        crate::catalog::SourcePreparation::Open(writer) => writer,
    };
    transaction::read_session(store.root(), &request.operation_id)?;
    let dir = transaction::candidate_dir(store.root(), &request.operation_id);
    for (requested, (plan_hex, _)) in request.profiles.iter().zip(plans) {
        if !dir.join(format!("convert-{plan_hex}.jsonl")).is_file() {
            continue;
        }
        let journal = super::journal::Journal::open(&dir, &plan_hex)?;
        if journal.is_empty() {
            continue;
        }
        for fence in &fences {
            fence.check()?;
        }
        durability::checkpoint(
            store,
            &durability::Policy {
                chain: &format!("{}/{}", request.operation_id, requested.slot),
                operation: &request.operation_id,
                plan: &format!("sha256:{plan_hex}"),
                interval: durability::INTERVAL_BYTES,
            },
            None,
            &journal,
        )?;
    }
    source_checkpoints(store, &request.operation_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::ingest::fingerprint::{Provenance, REAL};
    use crate::storage::Census;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-model-source-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn header_prefix(bytes: &[u8]) -> Vec<u8> {
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        bytes[..length + 8].to_vec()
    }

    fn carrier(path: &Path, key: &str, value: [u8; 4]) -> Vec<u8> {
        let header =
            format!("{{{key:?}:{{\"data_offsets\":[0,4],\"dtype\":\"F32\",\"shape\":[1]}}}}");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&value);
        fs::write(path, &bytes).unwrap();
        bytes
    }

    fn banked(
        component: &str,
        member: &str,
        path: &Path,
        profile: &str,
        encoding: &str,
    ) -> (Banked, SourceProfile) {
        let (header, raw) = read_carrier_at(path).unwrap();
        let observed = fingerprint::fingerprint(component, &header).unwrap();
        let entry = Banked {
            keyset_digest: observed.keyset_digest.clone(),
            dialect: "safetensors.diffusers".into(),
            component: component.into(),
            converter: "diffusers.identity/1".into(),
            tensor_schema_digest: observed.tensor_schema_digest,
            logical_keys: observed.logical_keys as u64,
            provenance: Provenance {
                kind: REAL.into(),
                source: "synthetic full bytes".into(),
                source_sha256: Some(
                    source_digest(&header, &raw)
                        .trim_start_matches("sha256:")
                        .to_string(),
                ),
                note: "source facade transaction fixture".into(),
            },
        };
        let source_profile = SourceProfile {
            auto_select: true,
            grammar: None,
            name: profile.into(),
            components: vec![SourceProfileComponent {
                component: component.into(),
                source_member: Some(member.into()),
                source_member_prefix: None,
                target_encoding: encoding.into(),
                variants: vec![super::fingerprint::SourceProfileVariant {
                    keyset_digest: observed.keyset_digest,
                    construction_order: vec!["weight".into()],
                }],
            }],
        };
        (entry, source_profile)
    }

    fn fixture(root: &Path, bad_second_encoding: bool) -> (Store, Vec<u8>, PrepareModelSource) {
        fixture_sized(root, bad_second_encoding, None)
    }

    /// `floats` present makes every tensor a REAL CAS object rather than an inline body.
    /// The inline threshold is 256 B, so a request built from four-byte scalars publishes
    /// nothing at all and any assertion about a cache over it would pass vacuously.
    fn fixture_sized(
        root: &Path,
        bad_second_encoding: bool,
        floats: Option<usize>,
    ) -> (Store, Vec<u8>, PrepareModelSource) {
        fs::create_dir_all(root).unwrap();
        let first_path = root.join("first.safetensors");
        let second_path = root.join("second.safetensors");
        let (first, second) = match floats {
            None => (
                carrier(&first_path, "weight", [1, 2, 3, 4]),
                carrier(&second_path, "weight", [5, 6, 7, 8]),
            ),
            Some(floats) => (
                wide_carrier(&first_path, "weight", floats, 0x11),
                wide_carrier(&second_path, "weight", floats, 0x22),
            ),
        };
        let (first_entry, first_profile) = banked(
            "encoder",
            "provider/first.safetensors",
            &first_path,
            "fixture/first/1",
            "plain/1",
        );
        let (second_entry, second_profile) = banked(
            "decoder",
            "provider/second.safetensors",
            &second_path,
            "fixture/second/1",
            if bad_second_encoding {
                "missing/1"
            } else {
                "plain/1"
            },
        );
        let mut registry = FingerprintRegistry::default();
        registry.merge(first_entry);
        registry.merge(second_entry);
        registry.merge_source_profile(first_profile);
        registry.merge_source_profile(second_profile);
        let store = Store::ensure(&root.join("store")).unwrap();
        let request = PrepareModelSource {
            operation_id: "model-source-fixture".into(),
            source_selection_digest: format!("sha256:{}", "11".repeat(32)),
            profiles: vec![
                ModelSourceProfile {
                    slot: "first".into(),
                    profile: "fixture/first/1".into(),
                },
                ModelSourceProfile {
                    slot: "second".into(),
                    profile: "fixture/second/1".into(),
                },
            ],
            roster: vec![
                VerifiedModelSourceFile {
                    member: "provider/first.safetensors".into(),
                    object_id: ObjectRef::of(&first).id(),
                    length: first.len() as u64,
                    path: first_path,
                    header: header_prefix(&first),
                },
                VerifiedModelSourceFile {
                    member: "provider/second.safetensors".into(),
                    object_id: ObjectRef::of(&second).id(),
                    length: second.len() as u64,
                    path: second_path,
                    header: header_prefix(&second),
                },
            ],
            landed: vec![
                "provider/first.safetensors".into(),
                "provider/second.safetensors".into(),
            ],
            checkpoints: Vec::new(),
            adopt_from_operation_id: None,
        };
        (store, registry.to_bytes(), request)
    }

    fn selected(request: &PrepareModelSource) -> Vec<SelectedSourceMember> {
        request
            .roster
            .iter()
            .map(|row| SelectedSourceMember {
                member: row.member.clone(),
                object: ObjectRef {
                    sha256: row.object_id.trim_start_matches("sha256:").into(),
                    length: row.length,
                },
                header: row.header.clone(),
            })
            .collect()
    }

    fn land(store: &Store, path: &Path) {
        let body = fs::read(path).unwrap();
        store
            .put_stream(
                &mut body.as_slice(),
                Some(&ObjectRef::of(&body)),
                &Default::default(),
            )
            .unwrap();
    }

    #[test]
    fn custodied_output_leaves_the_disk_and_is_never_converted_again() {
        use crate::ingest::custody;
        let root = temporary("custodied-selection");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let members = selected(&request);
        let operation = "bounded-selection";
        let prepare = |budget| {
            prepare_selected_source(
                &store,
                operation,
                &request.source_selection_digest,
                request.profiles.clone(),
                &members,
                "fixture:registry",
                &registry,
                budget,
            )
            .unwrap()
        };
        let (nothing, landed) = prepare(Some(0));
        assert!(landed.is_empty() && !nothing.complete && nothing.converted_roles == 0);
        assert!(
            nothing.spent.is_empty(),
            "no op is journalled, so no body is spent"
        );

        land(&store, &request.roster[0].path);
        let (first, landed) = prepare(Some(1 << 20));
        assert_eq!(landed, vec!["provider/first.safetensors".to_string()]);
        assert_eq!((first.complete, first.converted_roles), (false, 1));
        assert_eq!(first.spent, vec!["provider/first.safetensors".to_string()]);
        assert!(
            first.checkpoints.is_empty(),
            "custodied work writes no chain"
        );

        let journalled = custody::journalled(store.root(), operation).unwrap();
        assert!(!journalled.is_empty());
        assert_eq!(
            custody::record(&store, operation, &[ObjectRef::of(b"foreign")], "hub:p1")
                .unwrap_err()
                .code,
            Code::OBJECT_ID_MISMATCH,
        );
        custody::record(&store, operation, &journalled, "hub:p1").unwrap();
        let collected = crate::gc::collect(store.root(), false).unwrap();
        assert!(collected.reclaimed_bytes > 0);
        for object in &journalled {
            assert!(
                !store.contains(&object.sha256),
                "custodied output stays on disk"
            );
        }

        land(&store, &request.roster[1].path);
        let (second, _) = prepare(Some(1 << 20));
        assert!(second.complete);
        assert_eq!((second.converted_roles, second.resumed_roles), (1, 1));
        assert_eq!(second.sources.len(), 2);
        // Sparse held manifests: GC walks them without their custodied bodies.
        crate::gc::collect(store.root(), false).unwrap();
        let (again, _) = prepare(Some(0));
        assert!(again.replayed && again.complete);
        assert_eq!(again.sources, second.sources);
    }

    #[test]
    fn a_reordered_request_with_repeated_rows_is_the_same_request() {
        let root = temporary("request-order");
        let (store, registry, request) = fixture(&root, false);
        let mut shuffled = request.clone();
        shuffled.profiles.reverse();
        shuffled.profiles.push(shuffled.profiles[0].clone());
        shuffled.roster.reverse();
        shuffled.roster.push(shuffled.roster[1].clone());
        shuffled.landed.reverse();
        shuffled.landed.push(shuffled.landed[0].clone());
        let first =
            prepare_model_source_with_registry(&store, &shuffled, "fixture:registry", &registry)
                .unwrap();
        assert!(first.complete && !first.replayed);
        let replay =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.sources, first.sources);

        let mut conflict = request.clone();
        conflict.operation_id = "model-source-conflict".into();
        let mut other = conflict.profiles[0].clone();
        other.profile = "fixture/second/1".into();
        conflict.profiles.push(other);
        assert_eq!(
            prepare_model_source_with_registry(&store, &conflict, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::DUPLICATE_KEY
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_bound_registry_ingests_models_the_build_never_banked_at_any_precision() {
        let root = temporary("bound-registry");
        let (store, registry, mut request) = fixture(&root, false);
        // Banked from F32 carriers; the first carrier is now re-uploaded as F16.
        let first_path = request.roster[0].path.clone();
        let header = "{\"weight\":{\"data_offsets\":[0,2],\"dtype\":\"F16\",\"shape\":[1]}}";
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0x00, 0x3c]);
        fs::write(&first_path, &bytes).unwrap();
        request.roster[0].object_id = ObjectRef::of(&bytes).id();
        request.roster[0].length = bytes.len() as u64;
        request.roster[0].header = header_prefix(&bytes);

        // Unknown to the compiled-in registry until the reviewed data is bound.
        assert_eq!(
            prepare_model_source(&store, &request).unwrap_err().code,
            Code::UNREGISTERED_FINGERPRINT
        );
        bind_registry(&store, Some(&registry)).unwrap();
        let prepared = prepare_model_source(&store, &request).unwrap();
        assert!(prepared.complete);
        assert_eq!(prepared.sources.len(), 2);
        // Registry data without provenance is not reviewed data.
        assert!(bind_registry(&store, Some(b"{\"entries\":[{\"component\":\"x\"}]}")).is_err());
        bind_registry(&store, None).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore = "subprocess for source journal SIGKILL proof"]
    fn source_tail_snapshot_worker() {
        let root =
            PathBuf::from(std::env::var_os("TFS_SOURCE_TAIL_FIXTURE").expect("fixture root"));
        let (store, registry, mut request) = fixture_sized(&root, false, Some(1024));
        if std::env::var_os("TFS_SOURCE_TAIL_RAW_TREE").is_some() {
            let mut entries = Vec::new();
            for row in &request.roster {
                let reference = ObjectRef {
                    sha256: row.object_id[7..].into(),
                    length: row.length,
                };
                store
                    .put_file(&row.path, Some(&reference), &crate::store::Fault::default())
                    .unwrap();
                entries.push((row.member.clone(), reference));
            }
            let tree = crate::manifest::Manifest::from_files(entries).unwrap();
            let owner = crate::ids::object_id(b"source-tail-raw-fixture");
            crate::source_artifact::create(&store, &owner, &tree).unwrap();
            let (identity, roster, landed) =
                crate::source_artifact::conversion_roster(&store, &owner).unwrap();
            request.source_selection_digest = identity.id();
            request.roster = roster;
            request.landed = landed;
            fs::write(root.join("native-registry.json"), &registry).unwrap();
            fs::write(
                root.join("tail-fixture.json"),
                canon::write(&Value::obj(vec![
                    ("source_owner", Value::str(owner)),
                    ("operation", Value::str(request.operation_id.clone())),
                ])),
            )
            .unwrap();
        }
        super::super::journal::RECORD_FAULT.with(|fault| {
            *fault.borrow_mut() = Some(crate::store::Fault {
                stage: Some("journal-record".into()),
                ready: Some(root.join("journal-ready")),
            });
        });
        prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
            .unwrap();
        panic!("journal fault did not pause the worker");
    }

    fn interrupted_source_tail() -> (PathBuf, Store, Vec<u8>, PrepareModelSource) {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        let root = temporary("source-tail-sigkill");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ingest::source::tests::source_tail_snapshot_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("TFS_SOURCE_TAIL_FIXTURE", &root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while !root.join("journal-ready").exists() {
            assert!(
                child.try_wait().unwrap().is_none(),
                "source worker exited before journal commit"
            );
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("source worker never reached its journal commit");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            source_checkpoints(&store, &request.operation_id)
                .unwrap()
                .is_empty(),
            "first sub-4GiB journal group unexpectedly exported a head"
        );
        assert_eq!(
            snapshot_model_source(&store, &request, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::STORE_BUSY,
            "snapshot stole a live writer"
        );
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        (root, store, registry, request)
    }

    #[test]
    fn stopped_source_tail_snapshots_without_reconverting_and_adopts_independently() {
        let (root, store, registry, mut original) = interrupted_source_tail();
        // A missing first carrier makes any accidental re-conversion observable.
        fs::remove_file(&original.roster[0].path).unwrap();
        original.landed.remove(0);
        let heads =
            snapshot_model_source(&store, &original, "fixture:registry", &registry).unwrap();
        assert_eq!(heads.len(), 1);
        assert_eq!(heads[0].slot, "first");
        assert!(heads[0].head.bytes < durability::INTERVAL_BYTES);
        let head = heads[0].head.head.clone().unwrap();
        let chain = durability::local_chain(&store, &head, &heads[0].plan_digest).unwrap();
        let part = chain
            .blobs
            .iter()
            .find(|part| part.length == 4096)
            .unwrap()
            .clone();
        let before = transaction::read_session(store.root(), &original.operation_id).unwrap();
        for _ in 0..4 {
            assert_eq!(
                snapshot_model_source(&store, &original, "fixture:registry", &registry).unwrap(),
                heads
            );
        }
        assert_eq!(
            transaction::read_session(store.root(), &original.operation_id).unwrap(),
            before,
            "unchanged snapshots grew metadata roots"
        );
        let mut retry = original.clone();
        retry.operation_id = "source-tail-recipient".into();
        retry.adopt_from_operation_id = Some(original.operation_id.clone());
        let mut empty_recipient = retry.clone();
        empty_recipient.adopt_from_operation_id = None;
        empty_recipient.landed.clear();
        let empty = prepare_model_source_with_registry(
            &store,
            &empty_recipient,
            "fixture:registry",
            &registry,
        )
        .unwrap();
        assert!(!empty.complete && empty.checkpoints.is_empty());
        let completed =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(completed.complete);
        assert_eq!(completed.resumed_roles, 1);
        assert_eq!(completed.converted_roles, 1);
        assert_eq!(completed.converted_bytes, 4096);
        let control_root = temporary("source-tail-control");
        let (control_store, control_registry, control_request) =
            fixture_sized(&control_root, false, Some(1024));
        let control = prepare_model_source_with_registry(
            &control_store,
            &control_request,
            "fixture:registry",
            &control_registry,
        )
        .unwrap();
        assert_eq!(completed.sources, control.sources);
        release_model_source(&store, &original.operation_id).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        store.open_verified(&part.sha256).unwrap();
        for source in &completed.sources {
            validate_prepared_result(&store, source, &Default::default()).unwrap();
        }
        let replay =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(replay.replayed && replay.complete);
        assert_eq!(replay.converted_bytes, 0);
        assert_eq!(replay.sources, completed.sources);
        release_model_source(&store, &retry.operation_id).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        assert!(!store.contains(&part.sha256));
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(control_root).unwrap();
    }

    #[test]
    fn source_tail_adoption_keeps_own_profiles_and_acquires_missing_predecessor_profiles() {
        let root = temporary("source-tail-mixed-profiles");
        let (store, registry, original) = fixture_sized(&root, false, Some(1024));
        let first =
            prepare_model_source_with_registry(&store, &original, "fixture:registry", &registry)
                .unwrap();
        let mut retry = original.clone();
        retry.operation_id = "partially-adopted-recipient".into();
        retry.landed.pop();
        let partial =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(!partial.complete);
        assert_eq!(partial.checkpoints.len(), 1);
        fs::remove_file(&retry.roster[1].path).unwrap();
        retry.adopt_from_operation_id = Some(original.operation_id.clone());
        let adopted =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(adopted.complete);
        assert_eq!(adopted.converted_bytes, 0);
        assert_eq!(adopted.resumed_roles, 2);
        assert_eq!(adopted.sources, first.sources);
        assert_eq!(adopted.checkpoints[0], partial.checkpoints[0]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_tail_adoption_refuses_released_predecessor_even_when_bytes_remain() {
        let (root, store, registry, original) = interrupted_source_tail();
        let heads =
            snapshot_model_source(&store, &original, "fixture:registry", &registry).unwrap();
        let _bridge = crate::catalog::WriterGuard::acquire(store.root()).unwrap();
        release_model_source(&store, &original.operation_id).unwrap();
        let mut retry = original.clone();
        retry.operation_id = "released-predecessor-recipient".into();
        retry.adopt_from_operation_id = Some(original.operation_id.clone());
        retry.checkpoints = heads
            .into_iter()
            .map(|checkpoint| (checkpoint.slot, checkpoint.head.head.unwrap()))
            .collect();
        assert!(
            store.contains(&retry.checkpoints[0].1.sha256),
            "test needs still-resident snapshot bytes"
        );
        assert_eq!(
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::ROOT_ABSENT
        );
        assert!(
            !Catalog::open(store.root())
                .unwrap()
                .model_source_operations()
                .unwrap()
                .contains(&original.operation_id),
            "adoption recreated a released predecessor claim"
        );
        assert_eq!(
            snapshot_model_source(&store, &original, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::ROOT_ABSENT
        );
        drop(_bridge);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_tail_snapshot_refuses_prepared_metadata_without_its_root() {
        let root = temporary("source-tail-prepared-without-root");
        let (store, registry, original) = fixture_sized(&root, false, Some(1024));
        let prepared =
            prepare_model_source_with_registry(&store, &original, "fixture:registry", &registry)
                .unwrap();
        assert!(prepared.complete);
        fs::remove_file(
            transaction::candidate_dir(store.root(), &original.operation_id).join("session.json"),
        )
        .unwrap();
        assert_eq!(
            snapshot_model_source(&store, &original, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::ROOT_ABSENT
        );
        assert!(
            !transaction::candidate_dir(store.root(), &original.operation_id)
                .join("session.json")
                .exists()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_tail_snapshot_refuses_changed_work_and_missing_objects() {
        let (root, store, registry, original) = interrupted_source_tail();
        let dir = transaction::candidate_dir(store.root(), &original.operation_id);
        let mut journal_files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        journal_files.sort();
        assert_eq!(journal_files.len(), 1);
        let before = fs::read(&journal_files[0]).unwrap();
        let mut changed = original.clone();
        changed.source_selection_digest = format!("sha256:{}", "ff".repeat(32));
        assert_eq!(
            snapshot_model_source(&store, &changed, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );
        assert!(source_checkpoints(&store, &original.operation_id)
            .unwrap()
            .is_empty());
        let mut changed_plan = original.clone();
        changed_plan.profiles.pop();
        assert_eq!(
            snapshot_model_source(&store, &changed_plan, "fixture:registry", &registry)
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT,
            "snapshot accepted a changed closed profile plan",
        );
        assert_eq!(fs::read(&journal_files[0]).unwrap(), before);
        let objects = super::super::journal::session_objects(&dir).unwrap();
        assert_eq!(objects.len(), 1);
        fs::remove_file(store.blob_path(&objects[0].sha256)).unwrap();
        assert!(
            snapshot_model_source(&store, &original, "fixture:registry", &registry).is_err(),
            "missing journal object exported as usable work"
        );
        assert!(source_checkpoints(&store, &original.operation_id)
            .unwrap()
            .is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn edited_source_request_adopts_parts_with_independent_heads_and_roots() {
        let root = temporary("source-adoption");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let first =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(first.complete);
        let old_heads: Vec<_> = first
            .checkpoints
            .iter()
            .map(|item| (item.slot.clone(), item.head.head.clone().unwrap()))
            .collect();
        let old_links: Vec<_> = old_heads
            .iter()
            .map(|(_, head)| durability::local_link(&store, head).unwrap())
            .collect();
        let mut retry = request.clone();
        retry.operation_id = "edited-source-request".into();
        retry.checkpoints = old_heads.clone();
        retry.landed.clear();
        for row in &request.roster {
            fs::remove_file(&row.path).unwrap();
        }
        // Another request's heads are not this operation's: discarded, never adopted.
        let ordinary =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert_eq!(ordinary.resumed_roles, 0);
        assert!(!ordinary.complete);
        retry.adopt_from_operation_id = Some(request.operation_id.clone());
        let adopted =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(adopted.complete);
        assert_eq!(adopted.converted_bytes, 0);
        assert_eq!(adopted.converted_roles, 0);
        assert_eq!(adopted.sources, first.sources);
        for ((item, (_, old_head)), old_link) in
            adopted.checkpoints.iter().zip(&old_heads).zip(&old_links)
        {
            let head = item.head.head.as_ref().unwrap();
            assert_ne!(
                head, old_head,
                "new request must own a new checkpoint chain"
            );
            let new_link = durability::local_link(&store, head).unwrap();
            assert_eq!(
                new_link.operation,
                format!("{}/{}", retry.operation_id, item.slot)
            );
            assert_eq!(
                durability::local_link(&store, old_head).unwrap(),
                *old_link,
                "old chain changed"
            );
        }
        release_model_source(&store, &request.operation_id).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        let replay =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(replay.replayed && replay.complete);
        assert_eq!(replay.sources, adopted.sources);
        assert_eq!(replay.checkpoints, adopted.checkpoints);
        assert_eq!(replay.converted_bytes, 0);
        for source in &replay.sources {
            validate_prepared_result(&store, source, &Default::default()).unwrap();
        }
        release_model_source(&store, &retry.operation_id).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_source_adoption_resumes_remaining_work_after_predecessor_release() {
        let root = temporary("source-partial-adoption");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let first = prepare_model_source_with_budget(
            &store,
            &request,
            "fixture:registry",
            &registry,
            &mut transaction::Budget::new(1024),
            true,
        )
        .unwrap();
        assert!(!first.complete);
        assert_eq!(first.converted_roles, 1);
        let mut retry = request.clone();
        retry.operation_id = "edited-partial-source".into();
        retry.adopt_from_operation_id = Some(request.operation_id.clone());
        retry.checkpoints = first
            .checkpoints
            .iter()
            .map(|item| (item.slot.clone(), item.head.head.clone().unwrap()))
            .collect();
        retry.landed.clear();
        fs::remove_file(&retry.roster[0].path).unwrap();
        let adopted =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(!adopted.complete);
        assert_eq!(adopted.converted_bytes, 0);
        assert_eq!(adopted.resumed_roles, 1);
        release_model_source(&store, &request.operation_id).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        retry.landed.push(retry.roster[1].member.clone());
        let completed =
            prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry)
                .unwrap();
        assert!(completed.complete);
        assert_eq!(completed.converted_roles, 1);
        assert_eq!(completed.converted_bytes, 4096);
        assert_eq!(completed.resumed_roles, 1);
        assert_eq!(completed.checkpoints[0], adopted.checkpoints[0]);
        release_model_source(&store, &retry.operation_id).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_adoption_discards_what_it_cannot_continue_and_refuses_foreign_authority() {
        for arm in ["selection", "carrier", "operation", "slot", "missing"] {
            let root = temporary(&format!("source-adoption-{arm}"));
            let (store, registry, request) = fixture_sized(&root, false, Some(1024));
            let first =
                prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                    .unwrap();
            let mut retry = request.clone();
            retry.operation_id = "edited-source-request".into();
            retry.adopt_from_operation_id = Some(request.operation_id.clone());
            retry.checkpoints = first
                .checkpoints
                .iter()
                .map(|item| (item.slot.clone(), item.head.head.clone().unwrap()))
                .collect();
            retry.landed.clear();
            match arm {
                "selection" => {
                    retry.source_selection_digest = format!("sha256:{}", "55".repeat(32))
                }
                "carrier" => retry.roster[0].object_id = format!("sha256:{}", "55".repeat(32)),
                "operation" => retry.adopt_from_operation_id = Some("unrelated-operation".into()),
                "slot" => {
                    let left = retry.checkpoints[0].1.clone();
                    retry.checkpoints[0].1 = retry.checkpoints[1].1.clone();
                    retry.checkpoints[1].1 = left;
                }
                "missing" => {
                    let head = first.checkpoints[0].head.head.as_ref().unwrap();
                    let chain =
                        durability::local_chain(&store, head, &first.checkpoints[0].plan_digest)
                            .unwrap();
                    let part = chain
                        .blobs
                        .iter()
                        .find(|object| object.length == 4096)
                        .unwrap();
                    fs::remove_file(store.object_path(&part.sha256)).unwrap();
                }
                _ => unreachable!(),
            }
            let result =
                prepare_model_source_with_registry(&store, &retry, "fixture:registry", &registry);
            match arm {
                // Another operation's bytes are never adoption authority.
                "operation" => assert_eq!(result.unwrap_err().code, Code::ROOT_ABSENT),
                // A stale or damaged predecessor chain is dropped; the missing work waits
                // for its carriers instead of refusing.
                _ => {
                    let prepared = result.unwrap_or_else(|error| panic!("{arm}: {error:?}"));
                    assert!(!prepared.complete, "{arm}");
                    if arm != "missing" {
                        assert_eq!(prepared.resumed_roles, 0, "{arm}");
                    }
                }
            }
            assert!(Catalog::open(store.root())
                .unwrap()
                .model_source_operations()
                .unwrap()
                .contains(&request.operation_id));
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn large_source_adoption_retains_journal_parts_without_expanding_the_session_root() {
        let root = temporary("source-adoption-large-roster");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("large.safetensors");
        let keys: Vec<_> = (0..1024).map(|i| format!("weight_{i:04}")).collect();
        let mut header = Vec::new();
        let mut payload = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            let start = payload.len();
            for _ in 0..128 {
                payload.extend_from_slice(&(index as u32).to_le_bytes());
            }
            header.push(format!(
                "{key:?}:{{\"data_offsets\":[{start},{}],\"dtype\":\"F32\",\"shape\":[128]}}",
                payload.len()
            ));
        }
        let header = format!("{{{}}}", header.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        fs::write(&path, &bytes).unwrap();
        let (entry, mut profile) = banked(
            "model",
            "provider/large.safetensors",
            &path,
            "fixture/large/1",
            "plain/1",
        );
        profile.components[0].variants[0].construction_order = keys;
        let mut registry = FingerprintRegistry::default();
        registry.merge(entry);
        registry.merge_source_profile(profile);
        let registry = registry.to_bytes();
        let store = Store::init(&root.join("store")).unwrap();
        let mut request = PrepareModelSource {
            operation_id: "large-source-original".into(),
            source_selection_digest: format!("sha256:{}", "12".repeat(32)),
            profiles: vec![ModelSourceProfile {
                slot: "source".into(),
                profile: "fixture/large/1".into(),
            }],
            roster: vec![VerifiedModelSourceFile {
                member: "provider/large.safetensors".into(),
                object_id: ObjectRef::of(&bytes).id(),
                length: bytes.len() as u64,
                path: path.clone(),
                header: header_prefix(&bytes),
            }],
            landed: vec!["provider/large.safetensors".into()],
            checkpoints: Vec::new(),
            adopt_from_operation_id: None,
        };
        let first =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(first.complete);
        assert_eq!(first.converted_roles, 1024);
        let original = request.operation_id.clone();
        request.operation_id = "large-source-edited".into();
        request.adopt_from_operation_id = Some(original.clone());
        request.checkpoints = first
            .checkpoints
            .iter()
            .map(|item| (item.slot.clone(), item.head.head.clone().unwrap()))
            .collect();
        request.landed.clear();
        fs::remove_file(path).unwrap();
        let adopted =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(adopted.complete);
        assert_eq!(adopted.sources, first.sources);
        assert_eq!(adopted.converted_bytes, 0);
        assert_eq!(adopted.converted_roles, 0);
        let candidate = transaction::read_session(store.root(), &request.operation_id).unwrap();
        assert!(
            candidate.candidates.len() < 100,
            "tensor parts belong in the journal, not a duplicate root roster"
        );
        release_model_source(&store, &original).unwrap();
        crate::gc::collect(store.root(), false).unwrap();
        validate_prepared_result(&store, &adopted.sources[0], &Default::default()).unwrap();
        let replay =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(replay.replayed && replay.complete);
        assert_eq!(replay.converted_bytes, 0);
        release_model_source(&store, &request.operation_id).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_source_pass_reports_body_admission_then_resumes_without_reconversion() {
        let root = temporary("strict-source-admission");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let advance = |limit| {
            prepare_model_source_with_registry_budget(
                &store,
                &request,
                "fixture:registry",
                &registry,
                Some(limit),
            )
            .unwrap()
        };
        for limit in [0, 4095] {
            let held = advance(limit);
            assert!(!held.complete);
            assert_eq!(held.converted_roles, 0);
            assert_eq!(held.converted_bytes, 0);
            assert_eq!(held.largest_op_bytes, 0);
            assert_eq!(held.required_write_bytes, 4096);
            assert_eq!(held.deferred_ops, 2);
        }
        let first = advance(4096);
        assert!(!first.complete);
        assert_eq!(first.converted_bytes, 4096);
        assert_eq!(first.converted_roles, 1);
        assert_eq!(first.required_write_bytes, 0);
        let held = advance(0);
        assert_eq!(held.converted_bytes, 0);
        assert_eq!(held.resumed_roles, 1);
        assert_eq!(held.required_write_bytes, 4096);
        assert_eq!(held.checkpoints, first.checkpoints);
        let completed = advance(4096);
        assert!(completed.complete);
        assert_eq!(completed.converted_bytes, 4096);
        assert_eq!(completed.resumed_roles, 1);
        let replay = advance(0);
        assert!(replay.replayed && replay.complete);
        assert_eq!(replay.sources, completed.sources);
        assert_eq!(replay.converted_bytes, 0);
        assert_eq!(replay.required_write_bytes, 0);
        let clean = Store::init(&root.join("clean")).unwrap();
        let expected =
            prepare_model_source_with_registry(&clean, &request, "fixture:registry", &registry)
                .unwrap();
        assert_eq!(completed.sources, expected.sources);
        release_model_source(&store, &request.operation_id).unwrap();
        release_model_source(&clean, &request.operation_id).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn one_pass_budget_is_shared_by_profiles_and_allows_one_oversized_op() {
        for limit in [1024, 4096] {
            let root = temporary("bounded-profiles");
            let (store, registry, request) = fixture_sized(&root, false, Some(1024));
            let first = prepare_model_source_with_budget(
                &store,
                &request,
                "fixture:registry",
                &registry,
                &mut transaction::Budget::new(limit),
                true,
            )
            .unwrap();
            assert!(!first.complete);
            assert_eq!(first.converted_roles, 1);
            assert_eq!(first.converted_bytes, 4096);
            assert_eq!(first.largest_op_bytes, 4096);
            assert_eq!(first.deferred_ops, 1, "one budget covers both profiles");
            let mut stalled = request.clone();
            stalled.landed.clear();
            let no_work = prepare_model_source_with_budget(
                &store,
                &stalled,
                "fixture:registry",
                &registry,
                &mut transaction::Budget::new(limit),
                true,
            )
            .unwrap();
            assert_eq!(no_work.converted_bytes, 0);
            assert_eq!(no_work.largest_op_bytes, 0);
            assert_eq!(no_work.checkpoints, first.checkpoints);
            let second = prepare_model_source_with_budget(
                &store,
                &request,
                "fixture:registry",
                &registry,
                &mut transaction::Budget::new(limit),
                true,
            )
            .unwrap();
            assert!(second.complete);
            assert_eq!(second.converted_roles, 1);
            assert_eq!(second.resumed_roles, 1);
            assert_eq!(second.converted_bytes, 4096);
            let clean = Store::init(&root.join("clean")).unwrap();
            let expected =
                prepare_model_source_with_registry(&clean, &request, "fixture:registry", &registry)
                    .unwrap();
            assert_eq!(second.sources, expected.sources);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn full_byte_multi_profile_transaction_replays_conflicts_rebuilds_and_releases_holds() {
        let root = temporary("transaction");
        let (store, registry, request) = fixture(&root, false);
        let first =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(!first.replayed);
        assert_eq!(
            first
                .sources
                .iter()
                .map(|row| row.slot.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        for source in &first.sources {
            let reference = ObjectRef {
                sha256: source.manifest_digest.trim_start_matches("sha256:").into(),
                length: source.manifest_length,
            };
            store.read_manifest(&reference).unwrap();
        }
        let catalog = Catalog::open(store.root()).unwrap();
        assert_eq!(
            catalog.model_source_operations().unwrap(),
            std::slice::from_ref(&request.operation_id)
        );
        let holds = catalog.held_keys().unwrap();
        assert!(!holds.is_empty());
        assert!(Census::open(store.root())
            .unwrap()
            .gc_plan(&holds)
            .unwrap()
            .is_empty());

        let replay =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.sources, first.sources);
        assert_eq!(replay.checkpoints, first.checkpoints);
        let mut bodies_gone = request.clone();
        bodies_gone.landed.clear();
        for row in &bodies_gone.roster {
            fs::remove_file(&row.path).unwrap();
        }
        let recovered_heads =
            prepare_model_source_with_registry(&store, &bodies_gone, "fixture:registry", &registry)
                .unwrap();
        assert_eq!(recovered_heads.checkpoints, first.checkpoints);
        assert_eq!(recovered_heads.converted_bytes, 0);
        // Remaining arms need their original carriers again.
        carrier(&request.roster[0].path, "weight", [1, 2, 3, 4]);
        carrier(&request.roster[1].path, "weight", [5, 6, 7, 8]);

        let mut changed = request.clone();
        changed.roster[0].object_id = format!("sha256:{}", "33".repeat(32));
        assert_eq!(
            prepare_model_source_with_registry(&store, &changed, "fixture:registry", &registry,)
                .unwrap_err()
                .code,
            Code::TRANSACTION_CONFLICT
        );

        fs::remove_file(Catalog::path(store.root())).unwrap();
        let rebuilt = Catalog::initialize(store.root()).unwrap();
        let projection = Census::open(store.root()).unwrap().projection().unwrap();
        rebuilt.replace_all(&projection).unwrap();
        // Catalog destruction loses operation coordination, not retained journal bytes.
        // Enumeration does not guess a source type from generic filesystem sessions.
        assert!(rebuilt.model_source_operations().unwrap().is_empty());
        let recovered =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(!recovered.replayed);
        assert_eq!(recovered.sources, first.sources);

        assert_eq!(
            Catalog::open(store.root())
                .unwrap()
                .model_source_operations()
                .unwrap(),
            std::slice::from_ref(&request.operation_id)
        );

        let session = transaction::candidate_dir(store.root(), &request.operation_id);
        let saved_session = root.join("saved-session");
        fs::rename(&session, &saved_session).unwrap();
        fs::write(&session, b"not a directory").unwrap();
        assert_eq!(
            release_model_source(&store, &request.operation_id)
                .unwrap_err()
                .code,
            Code::IO_FAILED
        );
        assert_eq!(
            Catalog::open(store.root())
                .unwrap()
                .model_source_operations()
                .unwrap(),
            std::slice::from_ref(&request.operation_id)
        );
        fs::remove_file(&session).unwrap();
        fs::rename(saved_session, &session).unwrap();

        release_model_source(&store, &request.operation_id).unwrap();
        release_model_source(&store, &request.operation_id).unwrap();
        assert!(Catalog::open(store.root())
            .unwrap()
            .model_source_operations()
            .unwrap()
            .is_empty());
        assert!(Catalog::open(store.root())
            .unwrap()
            .held_keys()
            .unwrap()
            .is_empty());
        assert!(!Census::open(store.root())
            .unwrap()
            .gc_plan(&[])
            .unwrap()
            .is_empty());
        for source in &first.sources {
            assert!(store
                .manifest_path(source.manifest_digest.trim_start_matches("sha256:"))
                .is_file());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn durable_source_heads_are_readable_without_restarting_conversion() {
        let root = temporary("source-head-readback");
        let (store, registry, request) = fixture_sized(&root, false, Some(256));
        let produced =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert_eq!(
            source_checkpoints(&store, &request.operation_id).unwrap(),
            produced.checkpoints
        );
        let reopened = Store::open(store.root()).unwrap();
        assert_eq!(
            source_checkpoints(&reopened, &request.operation_id).unwrap(),
            produced.checkpoints
        );
        release_model_source(&store, &request.operation_id).unwrap();
        assert_eq!(
            source_checkpoints(&store, &request.operation_id)
                .unwrap_err()
                .code,
            Code::ROOT_ABSENT
        );
        fs::remove_dir_all(root).unwrap();
    }

    // The replacement-Runtime consumer needs a real native prepared source and head,
    // not hand-authored SQL. Explicitly opt in; the caller owns the retained fixture.
    #[test]
    #[ignore = "exports an owned source fixture for cross-process Runtime restart proof"]
    fn export_source_operation_restart_fixture() {
        let root = PathBuf::from(
            std::env::var_os("TFS_SOURCE_RESTART_FIXTURE").expect("fixture path required"),
        );
        assert!(!root.exists(), "fixture export requires a fresh owned path");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        fs::write(root.join("native-registry.json"), &registry).unwrap();
        let prepared =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(prepared.complete);
        let heads = prepared
            .checkpoints
            .iter()
            .map(|checkpoint| {
                let head = checkpoint
                    .head
                    .head
                    .as_ref()
                    .expect("non-inline source exports a head");
                Value::obj(vec![
                    ("slot", Value::str(checkpoint.slot.clone())),
                    ("plan_digest", Value::str(checkpoint.plan_digest.clone())),
                    ("head_id", Value::str(head.id())),
                    ("head_length", Value::uint(head.length)),
                ])
            })
            .collect();
        let mut incomplete = request.clone();
        incomplete.operation_id = "open-source-operation".into();
        incomplete.landed.clear();
        let open =
            prepare_model_source_with_registry(&store, &incomplete, "fixture:registry", &registry)
                .unwrap();
        assert!(!open.complete);
        let document = Value::obj(vec![
            ("operation", Value::str(request.operation_id)),
            ("open_operation", Value::str(incomplete.operation_id)),
            (
                "store",
                Value::str(store.root().to_string_lossy().to_string()),
            ),
            ("heads", Value::arr(heads)),
        ]);
        fs::write(
            root.join("restart-fixture.json"),
            crate::canon::write(&document),
        )
        .unwrap();
    }

    #[test]
    fn a_bound_cache_does_not_block_or_claim_source_checkpoint_durability() {
        let root = temporary("bound-local-checkpoint");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let cache_root = root.join("repo-cache");
        let store = store.bind_repo_cache(Some(&cache_root)).unwrap();
        let existed = cache_root.exists();
        let prepared =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(prepared.complete);
        assert_eq!(prepared.checkpoints.len(), 2);
        assert!(prepared
            .checkpoints
            .iter()
            .all(|row| row.head.head.is_some() && row.head.bytes > 0));
        assert_eq!(
            cache_root.exists(),
            existed,
            "local checkpoint export must not touch NFS"
        );
        for checkpoint in &prepared.checkpoints {
            let link =
                durability::local_link(&store, checkpoint.head.head.as_ref().unwrap()).unwrap();
            assert_eq!(link.plan, checkpoint.plan_digest);
            assert!(link.progress.is_some());
        }
        let _ = fs::remove_dir_all(root);
    }

    /// The other half of the ruling: a binding that names a mount which is not there costs
    /// the conversion its resumability and NOTHING ELSE. Same request, same registry, same
    /// manifests — the run degrades to local-only in silence, exactly as an unbound Store
    /// does, because cache failure is allowed to change cost and never authority.
    #[test]
    fn a_binding_to_an_absent_mount_degrades_to_a_local_only_conversion() {
        let bound_root = temporary("absent-mount");
        let (store, registry, request) = fixture_sized(&bound_root, false, Some(1024));
        // A mount point that is NOT A DIRECTORY, which is what an unmounted volume looks
        // like from inside the container: every path under it fails to open and fails to be
        // created, for every object, in both directions.
        let stub = bound_root.join("never");
        fs::write(&stub, b"not a mount").unwrap();
        let absent = stub.join("mounted");
        let store = store.bind_repo_cache(Some(&absent)).unwrap();
        let degraded =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(
            degraded
                .checkpoints
                .iter()
                .all(|row| row.head.head.is_some()),
            "local progress must remain exportable when the cache is absent"
        );
        assert!(!absent.exists(), "a missing cache must not be conjured");

        let plain_root = temporary("no-mount");
        let (plain_store, plain_registry, plain_request) =
            fixture_sized(&plain_root, false, Some(1024));
        assert!(plain_store.repo_cache().is_none());
        let plain = prepare_model_source_with_registry(
            &plain_store,
            &plain_request,
            "fixture:registry",
            &plain_registry,
        )
        .unwrap();
        assert_eq!(
            degraded
                .sources
                .iter()
                .map(|s| (&s.slot, &s.manifest_digest, s.manifest_length))
                .collect::<Vec<_>>(),
            plain
                .sources
                .iter()
                .map(|s| (&s.slot, &s.manifest_digest, s.manifest_length))
                .collect::<Vec<_>>(),
            "the cache changed what the conversion produced"
        );
        for root in [bound_root, plain_root] {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn metadata_failure_exposes_no_result_or_output_holds() {
        let root = temporary("partial");
        let (store, registry, request) = fixture(&root, true);
        assert_eq!(
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry,)
                .unwrap_err()
                .code,
            Code::UNKNOWN_ENCODING
        );
        let digest = request_digest(&request, &registry);
        let catalog = Catalog::open(store.root()).unwrap();
        let open = catalog
            .begin_source_preparation(&request.operation_id, &digest)
            .unwrap();
        assert!(matches!(open, crate::catalog::SourcePreparation::Open(_)));
        drop(open);
        assert!(catalog.held_keys().unwrap().is_empty());
        release_model_source(&store, &request.operation_id).unwrap();
        assert!(catalog.held_keys().unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    // ------------------------------------------------------------ the conversion journal
    //
    // Integration, not unit: a real Store, a real plan resolved off real carrier headers,
    // real segmentation through the real CAS, and a real crash (the process's journal handle
    // is dropped mid-plan, exactly as a released pod drops everything it held).

    fn wide_carrier(path: &Path, key: &str, floats: usize, fill: u8) -> Vec<u8> {
        let bytes = floats * 4;
        let header = format!(
            "{{{key:?}:{{\"data_offsets\":[0,{bytes}],\"dtype\":\"F32\",\"shape\":[{floats}]}}}}"
        );
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend(std::iter::repeat_n(fill, bytes));
        fs::write(path, &out).unwrap();
        out
    }

    /// One reviewed profile over TWO carriers, so a plan spans two files and a per-file pass
    /// has something to defer.
    fn two_carrier_fixture(root: &Path) -> (Store, Vec<u8>, Vec<CarrierInput>) {
        fs::create_dir_all(root).unwrap();
        let first_path = root.join("first.safetensors");
        let second_path = root.join("second.safetensors");
        wide_carrier(&first_path, "weight", 1024, 0x11);
        wide_carrier(&second_path, "weight", 2048, 0x22);

        let mut registry = FingerprintRegistry::default();
        let mut components = Vec::new();
        for (component, member, path) in [
            ("encoder", "provider/first.safetensors", &first_path),
            ("decoder", "provider/second.safetensors", &second_path),
        ] {
            let (header, raw) = read_carrier_at(path).unwrap();
            let observed = fingerprint::fingerprint(component, &header).unwrap();
            registry.merge(Banked {
                keyset_digest: observed.keyset_digest.clone(),
                dialect: "safetensors.diffusers".into(),
                component: component.into(),
                converter: "diffusers.identity/1".into(),
                tensor_schema_digest: observed.tensor_schema_digest,
                logical_keys: observed.logical_keys as u64,
                provenance: Provenance {
                    kind: REAL.into(),
                    source: "synthetic full bytes".into(),
                    source_sha256: Some(
                        source_digest(&header, &raw)
                            .trim_start_matches("sha256:")
                            .to_string(),
                    ),
                    note: "conversion journal fixture".into(),
                },
            });
            components.push(SourceProfileComponent {
                component: component.into(),
                source_member: Some(member.into()),
                source_member_prefix: None,
                target_encoding: "plain/1".into(),
                variants: vec![super::fingerprint::SourceProfileVariant {
                    keyset_digest: observed.keyset_digest,
                    construction_order: vec!["weight".into()],
                }],
            });
        }
        registry.merge_source_profile(SourceProfile {
            auto_select: true,
            grammar: None,
            name: "journal/pair/1".into(),
            components,
        });
        let store = Store::init(&root.join("store")).unwrap();
        let carriers = vec![
            CarrierInput {
                member: Some("provider/first.safetensors".into()),
                path: first_path,
            },
            CarrierInput {
                member: Some("provider/second.safetensors".into()),
                path: second_path,
            },
        ];
        (store, registry.to_bytes(), carriers)
    }

    #[test]
    fn retained_raw_tree_conversion_keeps_inputs_and_charges_output_space() {
        let root = temporary("retained-raw-conversion");
        let (store, registry, mut request) = fixture_sized(&root, false, Some(256));
        let mut entries = Vec::new();
        for row in &request.roster {
            let reference = ObjectRef {
                sha256: row.object_id.trim_start_matches("sha256:").into(),
                length: row.length,
            };
            store
                .put_file(&row.path, Some(&reference), &crate::store::Fault::default())
                .unwrap();
            entries.push((row.member.clone(), reference));
        }
        let tree = crate::manifest::Manifest::from_files(entries).unwrap();
        let owner = crate::ids::object_id(b"raw source owner");
        crate::source_artifact::create(&store, &owner, &tree).unwrap();
        let (identity, roster, landed) =
            crate::source_artifact::conversion_roster(&store, &owner).unwrap();
        request.source_selection_digest = identity.id();
        request.roster = roster;
        request.landed = landed;
        let occupied = store.occupancy().unwrap();
        let constrained = store
            .clone()
            .bind_disk_budget(Some(occupied + 1023))
            .unwrap();
        let refusal = prepare_model_source_with_registry(
            &constrained,
            &request,
            "fixture:registry",
            &registry,
        )
        .unwrap_err();
        assert_eq!(refusal.code, Code::CAPACITY_EXHAUSTED);
        let store = constrained.bind_disk_budget(None).unwrap();
        let result =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(result.complete);
        assert!(!result.spent.is_empty());
        crate::gc::collect(store.root(), false).unwrap();
        for row in tree.entries() {
            assert!(store.record_valid(&row.1.blob().sha256).is_ok());
        }
        // A fresh converter request can start over while the original raw artifact survives.
        request.operation_id = "changed-converter-attempt".into();
        assert!(
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap()
                .complete
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// A SHARDED model, prepared end to end out of a content-addressed staging area — the
    /// production shape, with nothing about it mocked.
    ///
    /// Every carrier sits at `staging/<hex>`: flat, named by a digest, no extension, no
    /// directory, no siblings. Two components each own a `*.safetensors.index.json` naming
    /// two shards, and the two indexes are BYTE-IDENTICAL, so six members present five
    /// paths — H3's 48-into-47 exactly. Nothing but the member says which component an
    /// index belongs to or which shards its `weight_map` reaches.
    ///
    /// Before this, no sharded model could reach conversion by this route at all: the plan
    /// read the index as a safetensors carrier and refused on its first eight bytes, and had
    /// that been fixed alone, `read_sharded` would have looked for the shards beside a hex
    /// digest and found nothing.
    #[test]
    fn a_sharded_model_prepares_from_content_addressed_carriers() {
        let root = temporary("sharded-cas");
        let area = root.join("staging");
        fs::create_dir_all(&area).unwrap();

        fn tensor_bytes(key: &str, floats: usize, fill: u8) -> Vec<u8> {
            let bytes = floats * 4;
            let header = format!(
                "{{{key:?}:{{\"data_offsets\":[0,{bytes}],\"dtype\":\"F32\",\"shape\":[{floats}]}}}}"
            );
            let mut out = (header.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(header.as_bytes());
            out.extend(std::iter::repeat_n(fill, bytes));
            out
        }

        // The staging area IS content addressed: one file per distinct object, whatever the
        // members call it.
        let stage = |bytes: &[u8]| -> PathBuf {
            let path = area.join(crate::sha256::hex_digest(bytes));
            if !path.exists() {
                fs::write(&path, bytes).unwrap();
            }
            path
        };

        // One index document, shared verbatim by both components.
        let index_bytes = br#"{"weight_map":{"bias":"model-00002-of-00002.safetensors","weight":"model-00001-of-00002.safetensors"}}"#.to_vec();
        let mut files: Vec<VerifiedModelSourceFile> = Vec::new();
        let mut push = |member: &str, bytes: &[u8], path: PathBuf| {
            files.push(VerifiedModelSourceFile {
                member: member.to_string(),
                object_id: ObjectRef::of(bytes).id(),
                length: bytes.len() as u64,
                path,
                header: if member.ends_with(".index.json") {
                    bytes.to_vec()
                } else {
                    header_prefix(bytes)
                },
            });
        };
        for (directory, fill) in [("enc", 0x11u8), ("dec", 0x22u8)] {
            let weight = tensor_bytes("weight", 512, fill);
            let bias = tensor_bytes("bias", 256, fill ^ 0xFF);
            let index = stage(&index_bytes);
            push(
                &format!("{directory}/model.safetensors.index.json"),
                &index_bytes,
                index,
            );
            let staged = stage(&weight);
            push(
                &format!("{directory}/model-00001-of-00002.safetensors"),
                &weight,
                staged,
            );
            let staged = stage(&bias);
            push(
                &format!("{directory}/model-00002-of-00002.safetensors"),
                &bias,
                staged,
            );
        }
        files.sort_by(|left, right| left.member.cmp(&right.member));

        assert_eq!(files.len(), 6, "six verified members");
        let paths: std::collections::BTreeSet<&PathBuf> =
            files.iter().map(|file| &file.path).collect();
        assert_eq!(
            paths.len(),
            5,
            "the shared index collapses six members onto five files"
        );

        let carriers: Vec<CarrierInput> = files
            .iter()
            .map(|file| CarrierInput {
                member: Some(file.member.clone()),
                path: file.path.clone(),
            })
            .collect();
        let set = CarrierSet::of(&carriers);

        // Bank each component off the JOINED header its own member reaches.
        let mut registry = FingerprintRegistry::default();
        let mut components = Vec::new();
        for (component, directory) in [("encoder", "enc"), ("decoder", "dec")] {
            let member = format!("{directory}/model.safetensors.index.json");
            let carrier = CarrierInput {
                member: Some(member.clone()),
                path: set.path_of(&member).unwrap().to_path_buf(),
            };
            let (header, raw) = read_carrier(&carrier, &set).expect("a sharded carrier reads");
            assert_eq!(header.shards.len(), 2, "{member} joins two shards");
            for shard in &header.shards {
                assert_eq!(
                    shard.path.parent().unwrap(),
                    area,
                    "shards resolve inside the staging area, not beside a digest"
                );
            }
            let observed = fingerprint::fingerprint(component, &header).unwrap();
            registry.merge(Banked {
                keyset_digest: observed.keyset_digest.clone(),
                dialect: "safetensors.diffusers".into(),
                component: component.into(),
                converter: "diffusers.identity/1".into(),
                tensor_schema_digest: observed.tensor_schema_digest,
                logical_keys: observed.logical_keys as u64,
                provenance: Provenance {
                    kind: REAL.into(),
                    source: "synthetic full bytes".into(),
                    source_sha256: Some(
                        source_digest(&header, &raw)
                            .trim_start_matches("sha256:")
                            .to_string(),
                    ),
                    note: "sharded CAS fixture".into(),
                },
            });
            components.push(SourceProfileComponent {
                component: component.into(),
                source_member: Some(member),
                source_member_prefix: None,
                target_encoding: "plain/1".into(),
                variants: vec![super::fingerprint::SourceProfileVariant {
                    keyset_digest: observed.keyset_digest,
                    construction_order: header
                        .tensors
                        .iter()
                        .map(|tensor| tensor.key.clone())
                        .collect(),
                }],
            });
        }
        registry.merge_source_profile(SourceProfile {
            auto_select: true,
            grammar: None,
            name: "sharded/pair/1".into(),
            components,
        });
        let registry_bytes = registry.to_bytes();

        let store = Store::init(&root.join("store")).unwrap();
        let request = PrepareModelSource {
            operation_id: "sharded-cas".into(),
            source_selection_digest: format!("sha256:{}", "44".repeat(32)),
            profiles: vec![ModelSourceProfile {
                slot: "pair".into(),
                profile: "sharded/pair/1".into(),
            }],
            roster: files.clone(),
            landed: files.iter().map(|row| row.member.clone()).collect(),
            checkpoints: Vec::new(),
            adopt_from_operation_id: None,
        };

        let prepared = prepare_model_source_with_registry(
            &store,
            &request,
            "fixture:registry",
            &registry_bytes,
        )
        .expect("a sharded model prepares from staged carriers");
        assert!(!prepared.replayed);
        assert_eq!(prepared.sources.len(), 1);
        let reference = ObjectRef {
            sha256: prepared.sources[0]
                .manifest_digest
                .trim_start_matches("sha256:")
                .into(),
            length: prepared.sources[0].manifest_length,
        };
        let manifest = store.read_manifest(&reference).unwrap();
        crate::checkpoint::walk_cozytensors(&store, &manifest)
            .unwrap()
            .require_resident(&store)
            .expect("every converted tensor byte landed");

        // Only ONE physical shard exists on the next pod. Both index members still share
        // one object, and every other body is absent: metadata alone must plan all of them.
        let incremental = Store::init(&root.join("incremental")).unwrap();
        let body_root = root.join("incremental-bodies");
        fs::create_dir_all(&body_root).unwrap();
        let mut partial_request = request.clone();
        for row in &mut partial_request.roster {
            row.path = body_root.join(row.object_id.trim_start_matches("sha256:"));
        }
        let first_member = "enc/model-00001-of-00002.safetensors";
        let first_source = files.iter().find(|row| row.member == first_member).unwrap();
        let first_path = partial_request
            .roster
            .iter()
            .find(|row| row.member == first_member)
            .unwrap()
            .path
            .clone();
        fs::copy(&first_source.path, &first_path).unwrap();
        partial_request.landed = vec![first_member.into()];
        let partial = prepare_model_source_with_registry(
            &incremental,
            &partial_request,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert!(!partial.complete && partial.sources.is_empty());
        assert_eq!(
            partial.converted_roles, 1,
            "the arrived shard converts before its siblings"
        );
        assert_eq!(partial.deferred_ops, 3);
        assert!(partial.spent.contains(&first_member.to_string()));
        let progress = &partial.checkpoints[0];
        let head = progress.head.head.as_ref().unwrap();
        let link = durability::local_link(&incremental, head).unwrap();
        assert!(link.manifests.is_empty());
        assert!(link.progress.is_some());
        assert!(link.blobs.len() <= durability::CHECKPOINT_REFS);

        // Upload acknowledgment can lag multiple local advances. The next pass must extend
        // the locally exported chain, even when the owner still supplies no head or an old one.
        let bias_member = "enc/model-00002-of-00002.safetensors";
        let bias_source = files.iter().find(|row| row.member == bias_member).unwrap();
        let bias_path = partial_request
            .roster
            .iter()
            .find(|row| row.member == bias_member)
            .unwrap()
            .path
            .clone();
        fs::copy(&bias_source.path, &bias_path).unwrap();
        let mut unacknowledged = partial_request.clone();
        unacknowledged.landed.push(bias_member.into());
        let advanced = prepare_model_source_with_registry(
            &incremental,
            &unacknowledged,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert!(advanced.checkpoints[0].head.links > progress.head.links);
        assert_ne!(advanced.checkpoints[0].head.head, progress.head.head);
        unacknowledged.landed.clear();
        unacknowledged.checkpoints = vec![(progress.slot.clone(), head.clone())];
        let no_work = prepare_model_source_with_registry(
            &incremental,
            &unacknowledged,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert_eq!(no_work.converted_roles, 0);
        assert_eq!(
            no_work.checkpoints[0].head, advanced.checkpoints[0].head,
            "a delayed ACK cannot rewind the local cursor or append empty progress"
        );
        fs::remove_file(&first_path).unwrap();

        // Transfer the acknowledged chain through the ordinary admission door into a
        // different Store. No journal file, catalog row or source body is copied with it.
        let replacement = Store::init(&root.join("replacement")).unwrap();
        let mut delivered = Vec::new();
        let mut cursor = Some(head.clone());
        while let Some(object) = cursor {
            let mut bytes = Vec::new();
            incremental.read_into(&object.sha256, &mut bytes).unwrap();
            replacement
                .put_stream(&mut bytes.as_slice(), Some(&object), &Default::default())
                .unwrap();
            delivered.push(object.clone());
            let mut offset = 0;
            loop {
                let durability::Window {
                    link: page,
                    objects: refs,
                    next,
                } = durability::local_window(&incremental, &object, offset, 1).unwrap();
                assert!(
                    refs.len() <= 1,
                    "the caller can impose a smaller bounded page"
                );
                for (kind, object) in refs {
                    assert_eq!(kind, crate::repo_cache::CacheKind::Blob);
                    let mut bytes = Vec::new();
                    incremental.read_into(&object.sha256, &mut bytes).unwrap();
                    replacement
                        .put_stream(&mut bytes.as_slice(), Some(&object), &Default::default())
                        .unwrap();
                    delivered.push(object);
                }
                cursor = page.prev;
                if let Some(next) = next {
                    offset = next;
                } else {
                    break;
                }
            }
        }
        let next_bodies = root.join("replacement-bodies");
        let mut bootstrap = partial_request.clone();
        bootstrap.landed.clear();
        bootstrap.checkpoints = vec![(progress.slot.clone(), head.clone())];
        let bootstrapped = prepare_model_source_with_registry(
            &replacement,
            &bootstrap,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert_eq!(bootstrapped.checkpoints[0].head.head.as_ref(), Some(head));
        fs::create_dir_all(&next_bodies).unwrap();
        let mut next_request = partial_request.clone();
        next_request.landed.clear();
        next_request.checkpoints = vec![(progress.slot.clone(), head.clone())];
        for row in &mut next_request.roster {
            row.path = next_bodies.join(row.object_id.trim_start_matches("sha256:"));
            if row.member != first_member && row.member.ends_with(".safetensors") {
                let original = files
                    .iter()
                    .find(|original| original.member == row.member)
                    .unwrap();
                fs::copy(&original.path, &row.path).unwrap();
                next_request.landed.push(row.member.clone());
            }
        }
        // A changed immutable source cannot reuse the old journal even with equal headers:
        // the supplied chain is discarded and nothing is resumed from it.
        let mut changed = next_request.clone();
        changed.operation_id = "changed-plan".into();
        changed
            .roster
            .iter_mut()
            .find(|row| row.member == first_member)
            .unwrap()
            .object_id = format!("sha256:{}", "55".repeat(32));
        let reconverted = prepare_model_source_with_registry(
            &replacement,
            &changed,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert_eq!(reconverted.resumed_roles, 0);
        release_model_source(&replacement, &changed.operation_id).unwrap();
        let corrupt_store = Store::init(&root.join("corrupt-replacement")).unwrap();
        for object in &delivered {
            let mut bytes = Vec::new();
            replacement.read_into(&object.sha256, &mut bytes).unwrap();
            corrupt_store
                .put_stream(&mut bytes.as_slice(), Some(object), &Default::default())
                .unwrap();
        }
        let payload = link
            .blobs
            .iter()
            .find(|object| Some(*object) != link.progress.as_ref())
            .unwrap();
        fs::remove_file(corrupt_store.object_path(&payload.sha256)).unwrap();
        fs::write(
            corrupt_store.object_path(&payload.sha256),
            vec![0; payload.length as usize],
        )
        .unwrap();
        let corrupt = prepare_model_source_with_registry(
            &corrupt_store,
            &next_request,
            "fixture:registry",
            &registry_bytes,
        )
        .expect_err(
            "a checkpoint containing corrupt payload cannot certify locally complete progress",
        );
        assert!(matches!(
            corrupt.code,
            Code::OBJECT_ABSENT | Code::OBJECT_CORRUPT
        ));
        let mut no_ack = next_request.clone();
        no_ack.checkpoints.clear();
        let resumed = prepare_model_source_with_registry(
            &replacement,
            &no_ack,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert!(resumed.complete);
        assert_eq!(
            resumed.resumed_roles, 1,
            "the spent shard is never opened again"
        );
        assert_eq!(resumed.converted_roles, 3);
        assert_eq!(
            resumed.sources, prepared.sources,
            "interrupted and uninterrupted manifests are identical"
        );
        assert_eq!(resumed.checkpoints[0].plan_digest, progress.plan_digest);
        assert!(resumed.checkpoints[0].head.links > progress.head.links);
        let (holds, _) = crate::gc::session_holds(&replacement).unwrap();
        assert!(
            holds.iter().any(|held| held.sha256 == head.sha256),
            "the restored predecessor link stays held"
        );
        next_request.landed.clear();
        let replay = prepare_model_source_with_registry(
            &replacement,
            &next_request,
            "fixture:registry",
            &registry_bytes,
        )
        .unwrap();
        assert!(replay.replayed && replay.complete);
        assert_eq!(
            replay.checkpoints, resumed.checkpoints,
            "a lost final response is replayed even with an old ACK"
        );
        assert_eq!(replay.converted_bytes, 0);
        let mut unrelated = registry.clone();
        unrelated.entries[0]
            .provenance
            .note
            .push_str("; provenance correction only");
        assert!(
            prepare_model_source_with_registry(
                &replacement,
                &next_request,
                "fixture:registry",
                &unrelated.to_bytes(),
            )
            .unwrap()
            .replayed,
            "unrelated registry bytes do not change the actual plan"
        );

        // A selection short of one shard is refused BY MEMBER, and the refusal comes out of
        // the header pass — before a tensor byte moves.
        let mut short = request.clone();
        short.operation_id = "sharded-cas-short".into();
        let dropped = short
            .roster
            .iter()
            .position(|file| file.member == "enc/model-00002-of-00002.safetensors")
            .unwrap();
        let dropped = short.roster.remove(dropped);
        short.landed.retain(|member| member != &dropped.member);
        let refusal =
            prepare_model_source_with_registry(&store, &short, "fixture:registry", &registry_bytes)
                .expect_err("an index whose shard is not carried must refuse");
        assert_eq!(refusal.code, Code::MISSING_FIELD);
        assert!(
            refusal.detail.contains(&dropped.member),
            "the refusal names the missing member: {}",
            refusal.detail
        );

        let _ = fs::remove_dir_all(&root);
    }

    /// Two members carrying identical bytes resolve to ONE file, and the plan must accept
    /// that. A Store is content addressed, so a path is an OBJECT's identity and never a
    /// member's: MiniMax-H3's `FL2VA/transformer/model.safetensors.index.json` and its
    /// `Ref2VA/` counterpart are byte-identical, so its 48 verified members present 47
    /// distinct paths. Run 294 downloaded all 210.3 GB and refused here on DUPLICATE_KEY.
    #[test]
    fn two_members_may_share_one_carrier_path() {
        let root = temporary("shared-carrier-path");
        fs::create_dir_all(&root).unwrap();
        let shared = root.join("index.safetensors");
        wide_carrier(&shared, "weight", 512, 0x33);

        let mut registry = FingerprintRegistry::default();
        let mut components = Vec::new();
        // Two components, two member names, ONE file — the H3 shape exactly.
        for (component, member) in [("encoder", "FL2VA/index"), ("decoder", "Ref2VA/index")] {
            let (header, raw) = read_carrier_at(&shared).unwrap();
            let observed = fingerprint::fingerprint(component, &header).unwrap();
            registry.merge(Banked {
                keyset_digest: observed.keyset_digest.clone(),
                dialect: "safetensors.diffusers".into(),
                component: component.into(),
                converter: "diffusers.identity/1".into(),
                tensor_schema_digest: observed.tensor_schema_digest,
                logical_keys: observed.logical_keys as u64,
                provenance: Provenance {
                    kind: REAL.into(),
                    source: "synthetic full bytes".into(),
                    source_sha256: Some(
                        source_digest(&header, &raw)
                            .trim_start_matches("sha256:")
                            .to_string(),
                    ),
                    note: "shared carrier fixture".into(),
                },
            });
            components.push(SourceProfileComponent {
                component: component.into(),
                source_member: Some(member.into()),
                source_member_prefix: None,
                target_encoding: "plain/1".into(),
                variants: vec![super::fingerprint::SourceProfileVariant {
                    keyset_digest: observed.keyset_digest,
                    construction_order: vec!["weight".into()],
                }],
            });
        }
        registry.merge_source_profile(SourceProfile {
            auto_select: true,
            grammar: None,
            name: "shared/pair/1".into(),
            components,
        });

        let carriers = vec![
            CarrierInput {
                member: Some("FL2VA/index".into()),
                path: shared.clone(),
            },
            CarrierInput {
                member: Some("Ref2VA/index".into()),
                path: shared.clone(),
            },
        ];
        let (_, prepared) = plan_source_prepared(
            "registry",
            &registry.to_bytes(),
            &carriers,
            Some("shared/pair/1"),
            None,
        )
        .expect("two members sharing one path must plan");
        assert_eq!(prepared.files.len(), 2, "both members are planned");

        // The same row twice is one row; one member on two paths is a real conflict.
        let repeated = vec![
            carriers[0].clone(),
            carriers[1].clone(),
            carriers[0].clone(),
        ];
        let (_, again) = plan_source_prepared(
            "registry",
            &registry.to_bytes(),
            &repeated,
            Some("shared/pair/1"),
            None,
        )
        .expect("an identical repeated carrier row is merged");
        assert_eq!(again.files.len(), 2);
        let mut elsewhere = carriers[0].clone();
        elsewhere.path = root.join("elsewhere.safetensors");
        match plan_source_prepared(
            "registry",
            &registry.to_bytes(),
            &[carriers[0].clone(), carriers[1].clone(), elsewhere],
            Some("shared/pair/1"),
            None,
        ) {
            Err(refusal) => assert_eq!(refusal.code, Code::DUPLICATE_KEY),
            Ok(_) => panic!("one member on two paths must refuse"),
        }
        let _ = fs::remove_dir_all(root);
    }

    /// THE DECISIVE PROPERTY. A conversion interrupted partway resumes without re-hashing
    /// what it already converted, and produces the same Manifest it would have produced in
    /// one pass.
    ///
    /// Against the pre-journal `execute` this cannot pass: there was no record of a
    /// completed op at all, so every resumed run re-read and re-hashed the whole plan and
    /// only discovered the collision at the final link. It saved storage, never time.
    #[test]
    fn an_interrupted_conversion_resumes_without_rehashing_what_it_converted() {
        let root = temporary("resume-journal");
        let (store, registry, carriers) = two_carrier_fixture(&root);
        let (_, prepared) = plan_source_prepared(
            "fixture:registry",
            &registry,
            &carriers,
            Some("journal/pair/1"),
            None,
        )
        .unwrap();
        let specs = plain_specs(&prepared.seeds);
        // Which carrier a role reads is the PLAN's business (the profile's component order,
        // not the caller's carrier order), so the expectation is read off the plan.
        let bytes_from = |file_index: usize| -> u64 {
            prepared
                .plan
                .ops
                .iter()
                .flat_map(|op| op.roles.iter())
                .filter(|role| {
                    matches!(&role.bytes, convert::Bytes::Stream { file, .. } if *file == file_index)
                })
                .map(|role| crate::dtype::checked_bytes("role", &role.shape, role.dtype).unwrap())
                .sum()
        };
        let total = bytes_from(0) + bytes_from(1);
        assert_eq!(total, (1024 + 2048) * 4);

        // ---- the crash: only the first carrier has landed.
        let (_, writer) = transaction::open_root(&store, "journal-operation", "tenant").unwrap();
        let (partial, incomplete) = transaction::advance(
            &store,
            &prepared.plan,
            &prepared.files,
            &specs,
            &[],
            "journal-operation",
            transaction::Carriers::Landed(&[true, false]),
            None,
        )
        .unwrap();
        assert!(incomplete.is_none(), "no header exists over part of a plan");
        assert!(!partial.complete());
        assert_eq!(partial.deferred_ops, 1);
        assert_eq!(partial.converted_roles, 1);
        assert_eq!(partial.hashed, bytes_from(0));

        // The half-converted output is named on the filesystem, so a GC pass keeps it. The
        // session has no candidates at all yet — the header and Manifest are written last.
        let session_dir = transaction::candidate_dir(store.root(), "journal-operation");
        assert!(!super::super::journal::session_objects(&session_dir)
            .unwrap()
            .is_empty());
        let (holds, sessions) = crate::gc::session_holds(&store).unwrap();
        assert_eq!(sessions, ["journal-operation"]);
        assert!(!holds.is_empty());
        let survivors: Vec<String> = Census::open(store.root())
            .unwrap()
            .gc_plan(&holds)
            .unwrap()
            .iter()
            .map(|row| row.key.clone())
            .collect();
        for hold in &holds {
            assert!(
                !survivors.contains(&hold.key),
                "gc planned to reclaim in-flight conversion output {}",
                hold.key
            );
        }

        // ---- a real reclamation pass runs between the crash and the resume.
        //
        // This is the red arm for the second hazard: `catalog.hold()` wrote a row per 64 MiB
        // segment and `gc::collect` never reads that table. Before the journal, a pass here
        // reclaimed every byte of the half-converted output and the resume had nothing left
        // to reuse. GC takes the exclusive recovery lock, so the dead writer goes first —
        // exactly the state a released pod leaves behind.
        drop(writer);
        let objects: Vec<PathBuf> = super::super::journal::session_objects(&session_dir)
            .unwrap()
            .iter()
            .map(|object| store.object_path(&object.sha256))
            .collect();
        assert!(!objects.is_empty());
        let report = crate::gc::collect(store.root(), false).unwrap();
        assert!(report.kept_objects > 0, "the journal held nothing");
        assert_eq!(report.reclaimed_blobs, 0);
        for path in &objects {
            assert!(path.is_file(), "gc reclaimed in-flight conversion output");
        }

        // ---- the resume: a different process, after the writer died.
        let (_, writer) = transaction::open_root(&store, "journal-operation", "tenant").unwrap();
        let (resumed, outcome) = transaction::advance(
            &store,
            &prepared.plan,
            &prepared.files,
            &specs,
            &[],
            "journal-operation",
            transaction::Carriers::All,
            None,
        )
        .unwrap();
        assert!(resumed.complete());
        assert_eq!(
            resumed.resumed_roles, 1,
            "the first carrier's op must come out of the journal"
        );
        assert_eq!(
            resumed.hashed,
            bytes_from(1),
            "a resume hashes ONLY what it had not already converted"
        );
        let outcome = outcome.expect("a complete journal finalizes");

        // ---- the same artifact a single uninterrupted pass would have produced.
        let clean = temporary("resume-journal-oneshot");
        let (fresh, registry2, carriers2) = two_carrier_fixture(&clean);
        let (_, prepared2) = plan_source_prepared(
            "fixture:registry",
            &registry2,
            &carriers2,
            Some("journal/pair/1"),
            None,
        )
        .unwrap();
        let (_, fresh_writer) = transaction::open_root(&fresh, "one-shot", "tenant").unwrap();
        let one_shot = transaction::execute(
            &fresh,
            &prepared2.plan,
            &prepared2.files,
            &plain_specs(&prepared2.seeds),
            &[],
            "one-shot",
            None,
        )
        .unwrap();
        assert_eq!(one_shot.manifest_ref, outcome.manifest_ref);
        assert_eq!(one_shot.header_ref, outcome.header_ref);
        assert_eq!(one_shot.get("bytes_hashed"), total as i64);
        assert_eq!(outcome.get("roles_resumed"), 1);

        drop(writer);
        drop(fresh_writer);
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(clean);
    }

    /// A journal that is not complete may not become an artifact: a header over 44 of 48
    /// shards is a structurally valid header for a smaller model.
    #[test]
    fn a_partial_journal_refuses_to_become_a_header() {
        let root = temporary("partial-journal");
        let (store, registry, carriers) = two_carrier_fixture(&root);
        let (_, prepared) = plan_source_prepared(
            "fixture:registry",
            &registry,
            &carriers,
            Some("journal/pair/1"),
            None,
        )
        .unwrap();
        let specs = plain_specs(&prepared.seeds);
        let headers: Vec<(PathBuf, &super::carrier::SourceHeader)> = prepared
            .files
            .iter()
            .map(|file| (file.path.clone(), &file.header))
            .collect();
        let (_, writer) = transaction::open_root(&store, "partial", "tenant").unwrap();
        let mut journal = super::super::journal::Journal::open(
            &transaction::candidate_dir(store.root(), "partial"),
            &super::super::journal::plan_digest(&prepared.plan, &headers),
        )
        .unwrap();
        transaction::convert(
            &store,
            &prepared.plan,
            &prepared.files,
            &mut journal,
            "partial",
            transaction::Carriers::Landed(&[true, false]),
            None,
        )
        .unwrap();
        let refusal = transaction::finalize(
            &store,
            transaction::Finalize {
                plan: &prepared.plan,
                journal: &journal,
                specs: &specs,
                configs: &[],
                operation: "partial",
                pass: &transaction::Progress::default(),
                elapsed_ms: 0,
                mirror: None,
            },
        )
        .unwrap_err();
        assert_eq!(refusal.code, Code::VERIFIER_OUTPUT_INCOMPLETE);
        drop(writer);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verified_file_fence_refuses_symlinks_and_same_length_replacement() {
        use std::os::unix::fs::symlink;

        let root = temporary("file-fence");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("source.safetensors");
        let bytes = carrier(&path, "weight", [1, 2, 3, 4]);
        let mut verified = VerifiedModelSourceFile {
            member: "provider/source.safetensors".into(),
            object_id: ObjectRef::of(&bytes).id(),
            length: bytes.len() as u64,
            path: path.clone(),
            header: header_prefix(&bytes),
        };
        let fence = FileFence::capture(&verified).unwrap();

        let replacement = root.join("replacement.safetensors");
        fs::write(&replacement, &bytes).unwrap();
        fs::rename(&replacement, &path).unwrap();
        assert_eq!(fence.check().unwrap_err().code, Code::OBJECT_ID_MISMATCH);

        let link = root.join("source-link.safetensors");
        symlink(&path, &link).unwrap();
        verified.path = link;
        assert_eq!(
            FileFence::capture(&verified).unwrap_err().code,
            Code::KEY_GRAMMAR
        );
        let _ = fs::remove_dir_all(root);
    }
}
