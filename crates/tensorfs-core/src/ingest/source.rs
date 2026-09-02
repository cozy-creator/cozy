//! Reviewed foreign-source planning and preparation shared by the CLI and Python facade.
//!
//! The network and its credentials end before this module. Inputs are verified immutable local
//! paths plus exact provider identities. TensorFS re-fences those paths, re-proves the reviewed
//! header/profile match, streams selected tensor bytes through the ordinary ingest/CAS pass, and
//! retains only exact CAS holds and typed SQLite result rows.

use std::fs;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use crate::canon::{self, Value};
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

pub fn read_carrier(path: &Path) -> Result<(SourceHeader, Vec<u8>)> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".index.json"))
    {
        return carrier::read_sharded(path);
    }
    if path
        .extension()
        .is_some_and(|extension| extension == "json")
    {
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

pub fn prepare_with_registry(
    target: &str,
    sources: &[(String, PathBuf)],
    registry: &FingerprintRegistry,
    spec_variant: Option<usize>,
) -> Result<Prepared> {
    let target = Target::parse(target)?;
    let seeds = specs_for(&target, spec_variant)?;
    let specs = plain_specs(&seeds);
    let (mut files, mut raws, mut verdicts, mut fingerprints) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (component, path) in sources {
        let (full, raw) = read_carrier(path)?;
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
    let mut paths: Vec<&Path> = carriers
        .iter()
        .map(|carrier| carrier.path.as_path())
        .collect();
    paths.sort();
    paths.dedup();
    let mut members: Vec<&str> = carriers
        .iter()
        .filter_map(|carrier| carrier.member.as_deref())
        .collect();
    let member_count = members.len();
    members.sort();
    members.dedup();
    if paths.len() != carriers.len() || members.len() != member_count {
        return refuse(
            Code::DUPLICATE_KEY,
            "carrier paths and reviewed members must be unique",
        );
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
    let mut headers = Vec::with_capacity(carriers.len());
    for carrier in carriers {
        headers.push(read_carrier(&carrier.path)?.0);
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
    let sources: Vec<(String, PathBuf)> = assignments
        .iter()
        .map(|assignment| {
            (
                assignment.component.clone(),
                carriers[assignment.carrier].path.clone(),
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
    let mut prepared = prepare_with_registry(&target, &sources, &registry, spec_variant)?;
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

fn execute_prepared(store: &Store, operation: &str, prepared: &Prepared) -> Result<ObjectRef> {
    let _ = golden_suite(prepared.conv)?;
    for (_, seed) in &prepared.seeds {
        if let Some(vectors) = &seed.vectors {
            vectors.check(&seed.spec)?;
        }
    }
    let predicted = tensor_schema_digest_of(&plan_tensor_schema(&prepared.plan));
    let outcome = transaction::execute(
        store,
        &prepared.plan,
        &prepared.files,
        &plain_specs(&prepared.seeds),
        &[],
        operation,
    )?;
    if outcome.header.tensor_schema_digest() != predicted {
        return refuse(
            Code::TOPOLOGY_MISMATCH,
            "source plan and executed tensor schema differ",
        );
    }
    Ok(outcome.manifest_ref)
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
    let mut paths: Vec<&Path> = request.files.iter().map(|row| row.path.as_path()).collect();
    paths.sort();
    paths.dedup();
    if paths.len() != request.files.len() {
        return refuse(
            Code::DUPLICATE_KEY,
            "verified source file paths must be unique",
        );
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
        let manifest = execute_prepared(store, &request.operation_id, &prepared)?;
        sources.push(PreparedModelSource {
            slot: requested.slot.clone(),
            profile: requested.profile.clone(),
            manifest_digest: manifest.id(),
            manifest_length: manifest.length,
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
    crate::catalog::Catalog::open(store.root())?.release_source_preparation(operation_id)
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
        let (header, raw) = read_carrier(path).unwrap();
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
        fs::create_dir_all(root).unwrap();
        let first_path = root.join("first.safetensors");
        let second_path = root.join("second.safetensors");
        let first = carrier(&first_path, "weight", [1, 2, 3, 4]);
        let second = carrier(&second_path, "weight", [5, 6, 7, 8]);
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
