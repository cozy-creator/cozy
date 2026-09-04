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
    self, Banked, Fingerprint, FingerprintRegistry, SourceProfile, SourceProfileComponent,
};
use super::transaction::{self, SourceFile};

pub const BUILTIN_REGISTRY: &str = "builtin:fingerprint-registry/1";
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

fn profile_assignments(
    profile: &SourceProfile,
    registry: &FingerprintRegistry,
    carriers: &[CarrierInput],
    headers: &[SourceHeader],
) -> Vec<Vec<SourceAssignment>> {
    let mut candidates = Vec::new();
    for component in &profile.components {
        let mut hits = Vec::new();
        for (variant_index, variant) in component.variants.iter().enumerate() {
            for entry in registry.entries.iter().filter(|entry| {
                entry.component == component.component
                    && entry.keyset_digest == variant.keyset_digest
            }) {
                let projected = entry.converter == "single_file.identity/1";
                for (index, header) in headers.iter().enumerate() {
                    let member_matches = match (
                        component.source_member.as_deref(),
                        component.source_member_prefix.as_deref(),
                        carriers[index].member.as_deref(),
                    ) {
                        (Some(expected), None, Some(observed)) => expected == observed,
                        (None, Some(prefix), Some(observed)) => observed.starts_with(prefix),
                        (None, None, None) => true,
                        _ => false,
                    };
                    if !member_matches {
                        continue;
                    }
                    let view = if projected {
                        match convert::single_file_component(&component.component, header) {
                            Ok(view) => view,
                            Err(_) => continue,
                        }
                    } else {
                        header.clone()
                    };
                    let observed = match fingerprint::fingerprint(&component.component, &view) {
                        Ok(observed) => observed,
                        Err(_) => continue,
                    };
                    if observed.keyset_digest == variant.keyset_digest
                        && observed.tensor_schema_digest == entry.tensor_schema_digest
                        && observed.logical_keys as u64 == entry.logical_keys
                    {
                        hits.push((index, projected, variant_index));
                    }
                }
            }
        }
        if hits.is_empty() {
            return Vec::new();
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
    solutions
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

/// `sources` carries each component's CARRIER, not its path: a path alone cannot say what
/// the file is, and this is the second place — after the plan's own header pass — that
/// would otherwise read a CAS-resident index as a safetensors file.
pub fn prepare_with_registry(
    target: &str,
    sources: &[(String, CarrierInput)],
    set: &CarrierSet,
    registry: &FingerprintRegistry,
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
        let projected = convert::single_file_component(component, &full).ok();
        let mut selected = None;
        if let Some(view) = projected {
            let observed = fingerprint::fingerprint(component, &view)?;
            if let Ok(banked) = registry.authorize(component, &observed) {
                selected = Some((view, observed, banked));
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
    let conv = convert::converter(names[0])?;
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
    let mut members: Vec<&str> = carriers
        .iter()
        .filter_map(|carrier| carrier.member.as_deref())
        .collect();
    let member_count = members.len();
    members.sort();
    members.dedup();
    if members.len() != member_count {
        return refuse(Code::DUPLICATE_KEY, "reviewed members must be unique");
    }
    for carrier in carriers {
        if let Some(member) = &carrier.member {
            fingerprint::validate_source_member(member)?;
        }
    }
    let registry = FingerprintRegistry::parse(registry_bytes)?;
    if let Some(name) = requested_profile {
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
    // plan's header pass and `prepare_with_registry`'s cannot disagree about which file a
    // `weight_map` value names.
    let set = CarrierSet::of(carriers);
    let mut headers = Vec::with_capacity(carriers.len());
    for carrier in carriers {
        headers.push(read_carrier(carrier, &set)?.0);
    }
    let mut matches = Vec::new();
    for profile in &registry.source_profiles {
        if requested_profile.is_some_and(|name| name != profile.name) {
            continue;
        }
        for assignments in profile_assignments(profile, &registry, carriers, &headers) {
            matches.push((profile, assignments));
            if matches.len() >= 2 {
                break;
            }
        }
        if matches.len() >= 2 {
            break;
        }
    }
    if matches.is_empty() {
        return refuse(
            Code::UNREGISTERED_FINGERPRINT,
            "carrier set matches no reviewed source profile",
        );
    }
    if matches.len() != 1 {
        let profiles: Vec<&str> = matches
            .iter()
            .map(|(profile, _)| profile.name.as_str())
            .collect();
        return refuse(
            Code::AMBIGUOUS_CLASSIFICATION,
            format!(
                "carrier headers admit multiple component assignments in profiles \
                 {profiles:?}; exact provider member/slot review is required"
            ),
        );
    }
    let (profile, mut assignments) = matches.remove(0);
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
    let order: Vec<(String, String)> = profile
        .components
        .iter()
        .flat_map(|component| {
            let assignment = assignments
                .iter()
                .find(|assignment| assignment.component == component.component)
                .expect("profile assignment covers every component");
            component.variants[assignment.variant]
                .construction_order
                .iter()
                .map(|key| (component.component.clone(), key.clone()))
        })
        .collect();
    let mut prepared = prepare_with_registry(&target, &sources, &set, &registry, spec_variant)?;
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

/// Convert one prepared profile, writing every object it admits through to `cache` as it
/// lands.
///
/// **The durability chain is per (operation, slot) and named that way because that is the
/// unit a resume replays.** `chain` carries the slot; two profiles of one request convert
/// different plans, and a single chain over both could not be validated against either. The
/// plan digest each link records is `ingest::journal::plan_digest` — the SAME identity the
/// conversion journal keys its file on, so the two planes cannot disagree about which
/// conversion this is.
fn execute_prepared(
    store: &Store,
    operation: &str,
    chain: &str,
    prepared: &Prepared,
) -> Result<(ObjectRef, durability::Head)> {
    let _ = golden_suite(prepared.conv)?;
    for (_, seed) in &prepared.seeds {
        if let Some(vectors) = &seed.vectors {
            vectors.check(&seed.spec)?;
        }
    }
    let predicted = tensor_schema_digest_of(&plan_tensor_schema(&prepared.plan));
    let headers: Vec<(PathBuf, &carrier::SourceHeader)> = prepared
        .files
        .iter()
        .map(|file| (file.path.clone(), &file.header))
        .collect();
    let plan_digest = super::journal::plan_digest(&prepared.plan, &headers);
    let policy = durability::Policy {
        chain,
        operation,
        plan: &format!("sha256:{plan_digest}"),
        interval: durability::INTERVAL_BYTES,
    };
    let (converted, durable) = durability::with(store, &policy, |mirror| {
        transaction::execute(
            store,
            &prepared.plan,
            &prepared.files,
            &plain_specs(&prepared.seeds),
            &[],
            operation,
            mirror,
        )
    });
    let outcome = converted?;
    if outcome.header.tensor_schema_digest() != predicted {
        return refuse(
            Code::TENSOR_SCHEMA_MISMATCH,
            "source plan and executed tensor schema differ",
        );
    }
    Ok((outcome.manifest_ref, durable.journal))
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
}

/// One preparation request. It names WHAT to convert and nothing about where the output is
/// kept: the write-through cache is a property of the `Store` this runs against
/// ([`crate::store::REPO_CACHE_BINDING`]), so no caller of this type ever spells a mount
/// point.
///
/// That placement preserves the contract the field it replaced was documented with: a
/// conversion is the same conversion whether or not a volume happened to be bound, so a pod
/// with a cache and a pod without one replay each other's stored result rather than treating
/// it as a different request. The cache changes what a run COSTS and what survives it; it can
/// never change what it produces, which is why it is absent from [`request_digest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareModelSource {
    pub operation_id: String,
    pub source_selection_digest: String,
    pub profiles: Vec<ModelSourceProfile>,
    pub files: Vec<VerifiedModelSourceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelSource {
    pub slot: String,
    pub profile: String,
    pub manifest_digest: String,
    pub manifest_length: u64,
    /// Where this slot's durability chain ends. `head` is the digest the caller hands the
    /// hub, and `bytes` the monotonic watermark it compare-and-sets on. Empty on a replayed
    /// preparation and on a run with no cache: both are runs that published nothing.
    pub durable: durability::Head,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelSources {
    pub replayed: bool,
    pub sources: Vec<PreparedModelSource>,
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

fn validate_request(
    request: &PrepareModelSource,
    registry: &FingerprintRegistry,
) -> Result<Vec<FileFence>> {
    ascii_name(
        "model-source operation",
        &request.operation_id,
        crate::limits::MAX_NAME_BYTES,
    )?;
    prefixed("source selection digest", &request.source_selection_digest)?;
    if request.profiles.is_empty() || request.profiles.len() > 16 {
        return refuse(Code::COUNT_CAP, "source profile count is outside 1..=16");
    }
    let profile_keys: Vec<&str> = request
        .profiles
        .iter()
        .map(|row| row.slot.as_str())
        .collect();
    if profile_keys.windows(2).any(|window| window[0] >= window[1]) {
        return refuse(
            Code::SORT_ORDER,
            "source profiles must be sorted uniquely by slot",
        );
    }
    for row in &request.profiles {
        ascii_name("source slot", &row.slot, crate::limits::MAX_NAME_BYTES)?;
        if !registry
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
    if request.files.is_empty() || request.files.len() > 4096 {
        return refuse(
            Code::COUNT_CAP,
            "verified source file count is outside 1..=4096",
        );
    }
    let members: Vec<&str> = request
        .files
        .iter()
        .map(|row| row.member.as_str())
        .collect();
    if members.windows(2).any(|window| window[0] >= window[1]) {
        return refuse(
            Code::SORT_ORDER,
            "verified source files must be sorted uniquely by member",
        );
    }
    // Same rule as the plan above, stated where the object ids are still in hand: a path
    // is an OBJECT's identity, so it may repeat across members, but only ever bearing one
    // object. Two DIFFERENT objects on one path is a real fault and stays a refusal.
    let mut owners: BTreeMap<&Path, &str> = BTreeMap::new();
    for row in &request.files {
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
    request.files.iter().map(FileFence::capture).collect()
}

fn digest_field(hash: &mut crate::sha256::Sha256, value: &[u8]) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn request_digest(request: &PrepareModelSource, registry_bytes: &[u8]) -> String {
    let mut hash = crate::sha256::Sha256::new();
    digest_field(&mut hash, b"tensorfs.model-source-preparation");
    digest_field(&mut hash, &crate::sha256::digest(registry_bytes));
    digest_field(&mut hash, request.source_selection_digest.as_bytes());
    for row in &request.profiles {
        digest_field(&mut hash, row.slot.as_bytes());
        digest_field(&mut hash, row.profile.as_bytes());
    }
    for row in &request.files {
        digest_field(&mut hash, row.member.as_bytes());
        digest_field(&mut hash, row.object_id.as_bytes());
        digest_field(&mut hash, &row.length.to_le_bytes());
        digest_field(&mut hash, row.path.as_os_str().as_encoded_bytes());
    }
    format!("sha256:{}", crate::sha256::hex(&hash.finish()))
}

fn validate_prepared_result(store: &Store, source: &PreparedModelSource) -> Result<()> {
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
    crate::checkpoint::walk_cozytensors(store, &manifest)?.require_resident(store)?;
    Ok(())
}

fn prepare_model_source_with_registry(
    store: &Store,
    request: &PrepareModelSource,
    registry_locator: &str,
    registry_bytes: &[u8],
) -> Result<PreparedModelSources> {
    let registry = FingerprintRegistry::parse(registry_bytes)?;
    let fences = validate_request(request, &registry)?;
    let digest = request_digest(request, registry_bytes);
    let catalog = crate::catalog::Catalog::open(store.root())?;
    let writer = match catalog.begin_source_preparation(&request.operation_id, &digest)? {
        crate::catalog::SourcePreparation::Prepared(rows) => {
            if rows.len() != request.profiles.len()
                || rows
                    .iter()
                    .zip(request.profiles.iter())
                    .any(|(stored, requested)| {
                        stored.slot != requested.slot || stored.profile != requested.profile
                    })
            {
                return refuse(
                    Code::DURABILITY_UNPROVEN,
                    "stored source results differ from their exact request",
                );
            }
            let prepared = PreparedModelSources {
                replayed: true,
                sources: rows
                    .into_iter()
                    .map(|row| PreparedModelSource {
                        slot: row.slot,
                        profile: row.profile,
                        manifest_digest: format!("sha256:{}", row.manifest_sha256),
                        manifest_length: row.manifest_length,
                        durable: durability::Head::default(),
                    })
                    .collect(),
            };
            for source in &prepared.sources {
                validate_prepared_result(store, source)?;
            }
            return Ok(prepared);
        }
        crate::catalog::SourcePreparation::Open(writer) => writer,
    };
    let carriers: Vec<CarrierInput> = request
        .files
        .iter()
        .map(|file| CarrierInput {
            member: Some(file.member.clone()),
            path: file.path.clone(),
        })
        .collect();
    let mut sources = Vec::with_capacity(request.profiles.len());
    for requested in &request.profiles {
        for fence in &fences {
            fence.check()?;
        }
        let (_, prepared) = plan_source_prepared(
            registry_locator,
            registry_bytes,
            &carriers,
            Some(&requested.profile),
            None,
        )?;
        for source in &prepared.files {
            for shard in &source.header.shards {
                if !fences.iter().any(|fence| fence.path == shard.path) {
                    return refuse(
                        Code::OBJECT_ID_MISMATCH,
                        format!("profile opened unverified shard {}", shard.path.display()),
                    );
                }
            }
        }
        let chain = format!("{}/{}", request.operation_id, requested.slot);
        let (manifest, durable) =
            execute_prepared(store, &request.operation_id, &chain, &prepared)?;
        sources.push(PreparedModelSource {
            slot: requested.slot.clone(),
            profile: requested.profile.clone(),
            manifest_digest: manifest.id(),
            manifest_length: manifest.length,
            durable,
        });
        validate_prepared_result(store, sources.last().expect("just pushed source result"))?;
    }
    for fence in &fences {
        fence.check()?;
    }
    let rows: Vec<crate::catalog::SourcePreparationResult> = sources
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
    drop(writer);
    Ok(PreparedModelSources {
        replayed: false,
        sources,
    })
}

pub fn prepare_model_source(
    store: &Store,
    request: &PrepareModelSource,
) -> Result<PreparedModelSources> {
    prepare_model_source_with_registry(store, request, BUILTIN_REGISTRY, BUILTIN_REGISTRY_BYTES)
}

pub fn release_model_source(store: &Store, operation_id: &str) -> Result<()> {
    crate::catalog::Catalog::open(store.root())?.release_source_preparation(operation_id)?;
    // The conversion journal and the session root that carried it are this operation's, and
    // the caller has just said it is done with the result. Nothing may resume it afterwards,
    // and nothing should keep holding its objects alive.
    let _ = fs::remove_dir_all(transaction::candidate_dir(store.root(), operation_id));
    Ok(())
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
        let store = Store::init(&root.join("store")).unwrap();
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
            files: vec![
                VerifiedModelSourceFile {
                    member: "provider/first.safetensors".into(),
                    object_id: ObjectRef::of(&first).id(),
                    length: first.len() as u64,
                    path: first_path,
                },
                VerifiedModelSourceFile {
                    member: "provider/second.safetensors".into(),
                    object_id: ObjectRef::of(&second).id(),
                    length: second.len() as u64,
                    path: second_path,
                },
            ],
        };
        (store, registry.to_bytes(), request)
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

        let mut changed = request.clone();
        changed.files[0].object_id = format!("sha256:{}", "33".repeat(32));
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
        let recovered =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(!recovered.replayed);
        assert_eq!(recovered.sources, first.sources);

        release_model_source(&store, &request.operation_id).unwrap();
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

    /// THE DECISIVE ARM: a conversion driven through `prepare_model_source`, where NO
    /// CALLER NAMED A CACHE, publishes its output onto one anyway — because the Store it ran
    /// against is bound to it — and a second, entirely separate Store bound to the same
    /// mount reads the whole checkpoint back, digest-verified, off that cache alone.
    ///
    /// This is the arm the write-through plane was missing. `repo_cache::Mirror` merged able
    /// to publish and unreachable in production: the process that converts is the Runtime,
    /// and the pod fences the Runtime child from the cache root by name, so the converter
    /// could never be TOLD where the mount was. Nothing here tells it. Setup records the
    /// mount on the Store once, and every later handle — this one, and the fresh one below
    /// that has never met the first — finds it.
    ///
    /// The readback is not a file comparison. `restore_checkpoint` admits every object
    /// through `put_stream` at the digest that names it and finishes with the Store's own
    /// local closure walk, so "restored" here means exactly what it means after a pull.
    #[test]
    fn a_bound_store_mirrors_a_conversion_no_caller_named_a_cache_for() {
        let root = temporary("bound-mirror");
        let cache_root = root.join("repo-cache");
        let (store, registry, request) = fixture_sized(&root, false, Some(1024));
        let store = store.bind_repo_cache(Some(&cache_root)).unwrap();

        // The request carries no mount point, and there is no argument that could hold one.
        let prepared =
            prepare_model_source_with_registry(&store, &request, "fixture:registry", &registry)
                .unwrap();
        assert!(!prepared.replayed);
        assert!(
            prepared
                .sources
                .iter()
                .all(|source| source.durable.head.is_some() && source.durable.bytes > 0),
            "a bound Store's conversion published no resumable chain: {:?}",
            prepared.sources
        );

        // A SECOND POD. A brand new Store — no objects, no catalog rows, no shared handle —
        // bound to the same mount by the same one-line setup act, and opened FRESH so the
        // binding is proved to have come off disk rather than out of a variable.
        let elsewhere = root.join("second-pod");
        Store::init(&elsewhere)
            .unwrap()
            .bind_repo_cache(Some(&cache_root))
            .unwrap();
        let fresh = Store::open(&elsewhere).unwrap();
        assert_eq!(fresh.repo_cache().map(|c| c.root()), Some(&*cache_root));
        for source in &prepared.sources {
            let manifest = ObjectRef {
                sha256: source.manifest_digest.trim_start_matches("sha256:").into(),
                length: source.manifest_length,
            };
            assert!(
                fresh.read_manifest(&manifest).is_err(),
                "the second Store must start out unable to answer for this checkpoint"
            );
            let report = durability::restore_checkpoint(&fresh, "second-pod", &manifest).unwrap();
            assert!(
                report.complete,
                "the cache could not answer for {}: {report:?}",
                source.slot
            );
            assert!(report.admitted > 0, "nothing actually moved: {report:?}");
            validate_prepared_result(&fresh, source).unwrap();
        }

        // And an UNBOUND Store asking for a restore is a caller mistake, not weather: it is
        // told so rather than handed an empty success that reads like a cold cache.
        let unbound = Store::init(&root.join("unbound-pod")).unwrap();
        let manifest = ObjectRef {
            sha256: prepared.sources[0]
                .manifest_digest
                .trim_start_matches("sha256:")
                .into(),
            length: prepared.sources[0].manifest_length,
        };
        assert_eq!(
            durability::restore_checkpoint(&unbound, "unbound", &manifest)
                .unwrap_err()
                .code,
            Code::MISSING_FIELD
        );
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
                .sources
                .iter()
                .all(|source| source.durable.head.is_none() && source.durable.bytes == 0),
            "an absent mount reported durability it cannot honour: {:?}",
            degraded.sources
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
    fn partial_profile_failure_exposes_no_result_and_explicit_release_drops_its_holds() {
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
        assert!(!catalog.held_keys().unwrap().is_empty());
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
            files: files.clone(),
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

        // A selection short of one shard is refused BY MEMBER, and the refusal comes out of
        // the header pass — before a tensor byte moves.
        let mut short = request.clone();
        short.operation_id = "sharded-cas-short".into();
        let dropped = short
            .files
            .iter()
            .position(|file| file.member == "enc/model-00002-of-00002.safetensors")
            .unwrap();
        let dropped = short.files.remove(dropped);
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

        // The member rule still holds: the SAME member twice is a real fault.
        let repeated = vec![carriers[0].clone(), carriers[0].clone()];
        match plan_source_prepared(
            "registry",
            &registry.to_bytes(),
            &repeated,
            Some("shared/pair/1"),
            None,
        ) {
            Err(refusal) => assert_eq!(refusal.code, Code::DUPLICATE_KEY),
            Ok(_) => panic!("one member twice must refuse"),
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
