//! `tfs ingest` — the border, driven live (tfs-003).
//!
//! The split of duties here IS the boundary document, made executable:
//!
//!   `inspect` / `plan`   read HEADERS only. Zero tensor bytes, so every structural
//!                        refusal is decidable for the cost of a few hundred KiB.
//!                        `inspect --markers` is the ONE opt-in exception and it says so in
//!                        bytes: marker payloads ARE tensor bytes (tiny U8 JSON blobs), the
//!                        plan structurally cannot read them, and evidence is their home.
//!   `run`                spawns an ISOLATED child. The parent never parses the carrier.
//!   `ingest-worker`      the hermetic child: it parses, converts, writes candidates under
//!                        a temporary session, and emits a receipt. It cannot
//!                        install a manifest and it has no network stack linked into it.
//!   `install`            the COORDINATOR's act. It consumes the receipt, verifies the
//!                        binding, walks the tree, and alone promotes one root.
//!   `reap`               what happens to everything that never reached `install`.
//!
//! A candidate is never a release. Its temporary session is the visible difference.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tensorfs_core::canon::{self, as_arr, as_str, Fields, Value};
use tensorfs_core::checkpoint as ck;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::durability;
use tensorfs_core::err::Code;
use tensorfs_core::header::tensor_schema_digest_of;
use tensorfs_core::ids::{ObjectRef, Plain};
use tensorfs_core::ingest::carrier::{self, SourceHeader};
use tensorfs_core::ingest::convert::{self, Class, Effect, Rekey, Source, Target};
use tensorfs_core::ingest::fingerprint::{
    self, Banked, FingerprintRegistry, SourceProfile, SourceProfileComponent, SourceProfileVariant,
};
use tensorfs_core::ingest::fingerprint::{Provenance, REAL};
use tensorfs_core::ingest::source as source_core;
use tensorfs_core::ingest::stamp::{self, Stamp};
use tensorfs_core::ingest::transaction::{self as tx};
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::limits;
use tensorfs_core::registry;
use tensorfs_core::store::{Progress, Store};

use crate::watch::{self, Verdict, PROGRESS_PREFIX};
use crate::{bail, flag, flag_num, flag_on, rss_kb, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

// ------------------------------------------------------------------ shared plumbing

/// `--source component=path`, repeated. The component name comes from the CALLER, never
/// from the filename: a path is a hint about where bytes live and nothing else.
fn sources(f: &Flags) -> Result<Vec<(String, PathBuf)>, ExitCode> {
    let mut out = Vec::new();
    for (k, v) in f.iter() {
        if k == "source" {
            match v.split_once('=') {
                Some((c, p)) => out.push((c.to_string(), PathBuf::from(p))),
                None => {
                    eprintln!("REFUSED MISSING_FIELD: --source wants component=path, got {v:?}");
                    return Err(ExitCode::FAILURE);
                }
            }
        }
    }
    if out.is_empty() {
        eprintln!("REFUSED MISSING_FIELD: no --source component=path given");
        return Err(ExitCode::FAILURE);
    }
    out.sort();
    Ok(out)
}

/// Physical carriers for source-plan. A reviewed canonical member may disambiguate
/// equal headers; paths are locations only and never participate in classification.
type CarrierInput = source_core::CarrierInput;
type CarrierSet = source_core::CarrierSet;

/// `--source component=path` names a LOCATION and nothing else, so these carriers have no
/// member: the file name is the name and the directory is the shard scope. That is honest
/// for a checkout on disk, and it is exactly why `--source` cannot reach a sharded index
/// that lives in a store — there the name is a hex digest and there are no siblings. Use
/// `source-plan --carrier <member>=<path>` for that, which names members.
fn unnamed(sources: &[(String, PathBuf)]) -> Vec<(String, CarrierInput)> {
    sources
        .iter()
        .map(|(component, path)| {
            (
                component.clone(),
                CarrierInput {
                    member: None,
                    path: path.clone(),
                },
            )
        })
        .collect()
}

fn carriers(f: &Flags) -> Result<Vec<CarrierInput>, ExitCode> {
    let mut out = Vec::new();
    for (key, value) in f {
        if key == "carrier" {
            let (member, location) = match value.split_once('=') {
                Some((member, location)) => {
                    if let Err(error) = fingerprint::validate_source_member(member) {
                        return Err(bail(error));
                    }
                    (Some(member.to_string()), location)
                }
                None => (None, value.as_str()),
            };
            match fs::canonicalize(location) {
                Ok(path) => out.push(CarrierInput { member, path }),
                Err(error) => {
                    eprintln!("REFUSED IO_FAILED: resolve carrier {location:?}: {error}");
                    return Err(ExitCode::FAILURE);
                }
            }
        }
    }
    if out.is_empty() {
        eprintln!("REFUSED MISSING_FIELD: no --carrier <path> given");
        return Err(ExitCode::FAILURE);
    }
    let mut paths: Vec<&PathBuf> = out.iter().map(|carrier| &carrier.path).collect();
    paths.sort();
    paths.dedup();
    let mut members: Vec<&str> = out
        .iter()
        .filter_map(|carrier| carrier.member.as_deref())
        .collect();
    members.sort();
    let member_count = members.len();
    members.dedup();
    if paths.len() != out.len() || members.len() != member_count {
        eprintln!("REFUSED DUPLICATE_KEY: carrier paths and reviewed members must be unique");
        return Err(ExitCode::FAILURE);
    }
    Ok(out)
}

const BUILTIN_REGISTRY: &str = source_core::BUILTIN_REGISTRY;
const BUILTIN_REGISTRY_BYTES: &[u8] = source_core::BUILTIN_REGISTRY_BYTES;

pub(crate) fn registry_bytes(f: &Flags) -> Result<(String, Vec<u8>), ExitCode> {
    let Some(path) = flag(f, "registry") else {
        return Ok((
            BUILTIN_REGISTRY.to_string(),
            BUILTIN_REGISTRY_BYTES.to_vec(),
        ));
    };
    let path = match fs::canonicalize(path) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("REFUSED IO_FAILED: resolve source registry {path}: {error}");
            return Err(ExitCode::FAILURE);
        }
    };
    match fs::read(&path) {
        Ok(bytes) => Ok((path.to_string_lossy().to_string(), bytes)),
        Err(error) => {
            eprintln!(
                "REFUSED IO_FAILED: read source registry {}: {error}",
                path.display()
            );
            Err(ExitCode::FAILURE)
        }
    }
}

fn registry_bytes_at(locator: &str) -> Result<Vec<u8>, ExitCode> {
    if locator == BUILTIN_REGISTRY {
        return Ok(BUILTIN_REGISTRY_BYTES.to_vec());
    }
    match fs::read(locator) {
        Ok(bytes) => Ok(bytes),
        Err(error) => {
            eprintln!("REFUSED IO_FAILED: read source registry {locator}: {error}");
            Err(ExitCode::FAILURE)
        }
    }
}

pub(crate) fn load_registry(f: &Flags) -> Result<FingerprintRegistry, ExitCode> {
    let (locator, bytes) = registry_bytes(f)?;
    match FingerprintRegistry::parse(&bytes) {
        Ok(registry) => Ok(registry),
        Err(error) => {
            eprintln!(
                "REFUSED {}: {locator}: {}",
                error.code.as_str(),
                error.detail
            );
            Err(ExitCode::FAILURE)
        }
    }
}

/// Print the exact provider members named by reviewed source profiles. This
/// reads only the small registry and is the pre-body selection seam used by
/// callers that already hold provider metadata.
pub fn cmd_source_members(profiles: &[&str], f: &Flags) -> ExitCode {
    let registry = match load_registry(f) {
        Ok(registry) => registry,
        Err(code) => return code,
    };
    let profiles: Vec<String> = profiles
        .iter()
        .map(|profile| (*profile).to_string())
        .collect();
    let members = match registry.exact_source_members(&profiles) {
        Ok(members) => members,
        Err(error) => return bail(error),
    };
    for member in members {
        println!("{member}");
    }
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ marker evidence

/// One decoded `.comfy_quant` payload, JOINED to the companion roles its own module
/// supplies. The join is the point: the marker is the PRODUCER's claim about the bytes and
/// the roles are what the border actually selects on, so holding them side by side is the
/// only way a disagreement between the two becomes visible instead of assumed away.
type Marker = (String, fingerprint::ComfyQuant, Vec<&'static str>);

/// Read exactly the marker tensors' own bytes — nothing else in the data region is touched.
/// Bounded BEFORE allocation: a payload over the text cap refuses without being read.
fn marker_evidence(
    path: &Path,
    h: &SourceHeader,
) -> tensorfs_core::err::Result<(Vec<Marker>, u64)> {
    let (mut out, mut read) = (Vec::new(), 0u64);
    for t in &h.tensors {
        if !t.key.ends_with(".comfy_quant") {
            continue;
        }
        if t.nbytes() > limits::MAX_DOC_TEXT_BYTES as u64 {
            return tensorfs_core::err::refuse(
                tensorfs_core::err::Code::SIZE_CAP,
                format!(
                    "{}: marker payload declares {} B, over the {} B text cap — refused \
                     before it is read, not after",
                    t.key,
                    t.nbytes(),
                    limits::MAX_DOC_TEXT_BYTES
                ),
            );
        }
        let mut buf = Vec::with_capacity(t.nbytes() as usize);
        let (rp, abs) = carrier::resolve(path, h, t)?;
        carrier::Region::open(rp, abs, t.nbytes())?
            .read_to_end(&mut buf)
            .map_err(|e| tensorfs_core::err::Refusal {
                code: tensorfs_core::err::Code::IO_FAILED,
                detail: format!("{}: read marker payload: {e}", t.key),
            })?;
        read += buf.len() as u64;
        let q = fingerprint::decode_comfy_quant(&t.key, &buf)?;
        let module = &t.key[..t.key.len() - ".comfy_quant".len()];
        let roles: Vec<&'static str> = fingerprint::ROLE_SUFFIXES
            .iter()
            .filter(|s| h.get(&format!("{module}{s}")).is_some())
            .copied()
            .collect();
        out.push((t.key.clone(), q, roles));
    }
    Ok((out, read))
}

/// Print the decoded markers as typed evidence, grouped, plus the one cross-check worth
/// having: the producer says which modules keep a full-precision matmul, and the carrier's
/// own roles say which modules have no `.input_scale`. On the real fl2va DiT those are the
/// same 50 modules. A divergence would mean the marker and the bytes disagree — which is
/// exactly why selection reads the roles and never the marker.
fn print_markers(rows: &[Marker], read: u64) {
    let mut groups: Vec<(String, bool, Vec<&'static str>, usize)> = Vec::new();
    let mut divergent = 0usize;
    for (_, q, roles) in rows {
        if q.full_precision_matrix_mult == roles.contains(&".input_scale") {
            divergent += 1;
        }
        match groups
            .iter_mut()
            .find(|(f, b, r, _)| *f == q.format && *b == q.full_precision_matrix_mult && r == roles)
        {
            Some((_, _, _, n)) => *n += 1,
            None => groups.push((
                q.format.clone(),
                q.full_precision_matrix_mult,
                roles.clone(),
                1,
            )),
        }
    }
    groups.sort_by_key(|(_, _, _, n)| std::cmp::Reverse(*n));
    println!(
        "  markers      {} .comfy_quant payloads decoded, {read} tensor B read",
        rows.len()
    );
    for (fmt, fp, roles, n) in &groups {
        println!("    {n:>6}     format {fmt} · full_precision_matrix_mult {fp} · roles {roles:?}");
    }
    println!(
        "    cross-check  {} of {} agree with the carrier's own roles ({divergent} diverge) — \
         the border selects on the ROLES either way",
        rows.len() - divergent,
        rows.len()
    );
}

// ------------------------------------------------------------------ inspect

/// Hermetic bounded decode plus classification. The pickle / archive / GGUF refusals are
/// observed here, on the magic, with nothing interpreted.
pub fn cmd_inspect(f: &Flags) -> ExitCode {
    let srcs = match sources(f) {
        Ok(s) => s,
        Err(c) => return c,
    };
    let reg = load_registry(f).ok();
    let t0 = Instant::now();
    let (mut total_header, mut marker_bytes) = (0u64, 0u64);

    for (component, path) in &srcs {
        println!("{component}  {}", path.display());
        let (h, raw) = ok!(source_core::read_carrier_at(path));
        total_header += h.header_bytes;
        let fp = ok!(fingerprint::fingerprint(component, &h));
        let stamps = ok!(fingerprint::evidence_stamps(&h));
        println!(
            "  carrier      {} B header, data at {}, {} B file, {} B unclaimed",
            h.header_bytes, h.data_start, h.file_len, h.gap_bytes
        );
        println!(
            "  keys         {} logical + {} role + {} marker = {}",
            fp.logical_keys,
            fp.role_keys,
            fp.marker_keys,
            h.tensors.len()
        );
        let dt: Vec<String> = fp
            .dtypes
            .iter()
            .map(|(d, n)| format!("{} {n}", d.name()))
            .collect();
        println!("  dtypes       {}", dt.join(" · "));
        println!("  declared     {} B of tensor bytes", h.declared_bytes());
        println!("  keyset       {}", fp.keyset_digest);
        println!("  tensor schema {}", fp.tensor_schema_digest);
        println!("  source       {}", source_core::source_digest(&h, &raw));
        if !stamps.is_empty() {
            for (k, v) in &stamps {
                println!("  stamp        {k} = {v:?}");
            }
        }
        if flag_on(f, "markers") {
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                println!("  markers      (a header-only evidence file has no data region to read)");
            } else {
                let (rows, read) = ok!(marker_evidence(path, &h));
                marker_bytes += read;
                if rows.is_empty() {
                    println!("  markers      none");
                } else {
                    print_markers(&rows, read);
                }
            }
        }
        match &reg {
            None => println!("  verdict      (no registry loaded)"),
            Some(r) => match r.classify(&fp) {
                fingerprint::Verdict::Authorized(b) => println!(
                    "  verdict      AUTHORIZED {} via {} -> {}",
                    b.component, b.dialect, b.converter
                ),
                fingerprint::Verdict::ComponentAmbiguous { banked_as } => println!(
                    "  verdict      AMBIGUOUS — this key set is reviewed, but not as \
                     {component:?}; banked as {banked_as:?} — refuses"
                ),
                fingerprint::Verdict::Ambiguous(n) => {
                    println!("  verdict      AMBIGUOUS {n:?} — refuses")
                }
                fingerprint::Verdict::Unregistered => {
                    println!("  verdict      UNREGISTERED — refuses (a filename never authorizes)")
                }
            },
        }
    }
    println!(
        "read {total_header} B of headers in {:.1} ms, {} tensor bytes, peak RSS {:.1} MiB",
        t0.elapsed().as_secs_f64() * 1000.0,
        match marker_bytes {
            0 => "zero".to_string(),
            n => format!("{n} marker"),
        },
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ bank

/// Bank a reviewed registry entry. This is the REVIEW ACT, performed by a platform operator
/// against bytes they have in hand — never by an uploader and never from a filename.
pub fn cmd_bank(dialect: &str, converter: &str, f: &Flags) -> ExitCode {
    let srcs = match sources(f) {
        Ok(s) => s,
        Err(c) => return c,
    };
    let conv = ok!(convert::converter(converter));
    if conv.dialect != dialect {
        eprintln!(
            "REFUSED DIALECT_MISMATCH: converter {converter} serves {:?}, not {dialect:?}",
            conv.dialect
        );
        return ExitCode::FAILURE;
    }
    let Some(path) = flag(f, "registry") else {
        eprintln!(
            "REFUSED MISSING_FIELD: ingest bank requires an explicit --registry output; \
             the product registry is compiled into tfs"
        );
        return ExitCode::FAILURE;
    };
    let p = PathBuf::from(path);
    let mut reg = match fs::read(&p) {
        Ok(b) => ok!(FingerprintRegistry::parse(&b)),
        Err(_) => FingerprintRegistry::default(),
    };
    let mut reviewed = Vec::new();
    for (component, path) in &srcs {
        let (full, raw) = ok!(source_core::read_carrier_at(path));
        let h = if conv.dialect == "safetensors.single_file" {
            ok!(convert::single_file_component(conv, component, &full))
        } else {
            full
        };
        let fp = ok!(fingerprint::fingerprint(component, &h));
        let sd = source_core::source_digest(&h, &raw);
        reg.merge(Banked {
            keyset_digest: fp.keyset_digest.clone(),
            dialect: dialect.to_string(),
            component: component.clone(),
            converter: converter.to_string(),
            tensor_schema_digest: fp.tensor_schema_digest.clone(),
            logical_keys: fp.logical_keys as u64,
            provenance: Provenance {
                kind: REAL.to_string(),
                source: flag(f, "source-name")
                    .unwrap_or("local carrier")
                    .to_string(),
                source_sha256: Some(sd.trim_start_matches("sha256:").to_string()),
                note: format!(
                    "banked from a real carrier: {} logical keys, {} role keys, {} markers; \
                     header {} B; the source_sha256 is the carrier IDENTITY digest (header \
                     bytes + geometry), not a whole-file hash",
                    fp.logical_keys, fp.role_keys, fp.marker_keys, h.header_bytes
                ),
            },
        });
        println!(
            "banked {component}  {}  {}",
            fp.keyset_digest, fp.tensor_schema_digest
        );
        reviewed.push((component.clone(), fp.keyset_digest));
    }
    if let Some(name) = flag(f, "source-profile") {
        let order = match construction_order(f) {
            Ok(order) => order,
            Err(code) => return code,
        };
        let mut targets = Vec::new();
        let mut members = Vec::new();
        for (key, value) in f {
            if key != "target" && key != "member" {
                continue;
            }
            match value.split_once('=') {
                Some((component, encoding)) if !component.is_empty() && !encoding.is_empty() => {
                    if key == "target" {
                        targets.push((component.to_string(), encoding.to_string()));
                    } else {
                        members.push((component.to_string(), encoding.to_string()));
                    }
                }
                _ => {
                    eprintln!(
                        "REFUSED MISSING_FIELD: --target wants component=encoding, got {value:?}"
                    );
                    return ExitCode::FAILURE;
                }
            }
        }
        let mut component_names = Vec::new();
        for (component, _) in &order {
            if !component_names.contains(component) {
                component_names.push(component.clone());
            }
        }
        let mut components = Vec::new();
        for component in &component_names {
            let Some((_, keyset_digest)) = reviewed.iter().find(|(name, _)| name == component)
            else {
                eprintln!(
                    "REFUSED CONSTRUCTION_ORDER_REQUIRED: order names unbanked component {component:?}"
                );
                return ExitCode::FAILURE;
            };
            let hits: Vec<&str> = targets
                .iter()
                .filter(|(name, _)| name == component)
                .map(|(_, encoding)| encoding.as_str())
                .collect();
            if hits.len() != 1 {
                eprintln!(
                    "REFUSED MISSING_FIELD: source profile component {component:?} needs exactly one --target component=encoding"
                );
                return ExitCode::FAILURE;
            }
            components.push(SourceProfileComponent {
                component: component.clone(),
                source_member: match members
                    .iter()
                    .filter(|(name, _)| name == component)
                    .map(|(_, member)| member)
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [] => None,
                    [member] => Some((*member).clone()),
                    _ => {
                        eprintln!(
                            "REFUSED DUPLICATE_KEY: source profile component {component:?} has more than one --member"
                        );
                        return ExitCode::FAILURE;
                    }
                },
                source_member_prefix: None,
                target_encoding: hits[0].to_string(),
                variants: vec![SourceProfileVariant {
                    keyset_digest: keyset_digest.clone(),
                    construction_order: order
                        .iter()
                        .filter(|(name, _)| name == component)
                        .map(|(_, key)| key.clone())
                        .collect(),
                }],
            });
        }
        if components.len() != reviewed.len()
            || targets.len() != reviewed.len()
            || members
                .iter()
                .any(|(component, _)| !reviewed.iter().any(|(name, _)| name == component))
        {
            eprintln!(
                "REFUSED MISSING_FIELD: source profile order, targets, and banked components must name the same exact set"
            );
            return ExitCode::FAILURE;
        }
        let profile = SourceProfile {
            auto_select: true,
            name: name.to_string(),
            components,
        };
        let target = profile
            .components
            .iter()
            .map(|component| format!("{}={}", component.component, component.target_encoding))
            .collect::<Vec<_>>()
            .join(",");
        let variant = flag(f, "spec-variant").and_then(|value| value.parse().ok());
        let mut prepared = match source_core::prepare(
            &target,
            &unnamed(&srcs),
            &CarrierSet::default(),
            Some(&reg),
            variant,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return bail(error),
        };
        if let Err(error) = prepared.plan.apply_order(&order) {
            return bail(error);
        }
        reg.merge_source_profile(profile);
    } else if f
        .iter()
        .any(|(key, _)| matches!(key.as_str(), "target" | "member" | "order"))
    {
        eprintln!(
            "REFUSED MISSING_FIELD: --target/--order on ingest bank require --source-profile"
        );
        return ExitCode::FAILURE;
    }
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    // The REVIEW ACT does not get to leave behind a document the border cannot read. A
    // `--source-name` with one em dash in it wrote a registry that every later run refused
    // NON_ASCII_FIELD, at the far end of the pipeline and with the review already "done".
    let bytes = reg.to_bytes();
    ok!(FingerprintRegistry::parse(&bytes));
    if let Err(e) = fs::write(&p, &bytes) {
        eprintln!("REFUSED IO_FAILED: write {}: {e}", p.display());
        return ExitCode::FAILURE;
    }
    println!("{} entries -> {}", reg.entries.len(), p.display());
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ closed source plan

fn derive_source_plan(f: &Flags) -> Result<Value, ExitCode> {
    let carriers = carriers(f)?;
    let (registry_locator, registry_bytes) = registry_bytes(f)?;
    let variant = flag(f, "spec-variant").and_then(|value| value.parse().ok());
    source_core::plan_source(
        &registry_locator,
        &registry_bytes,
        &carriers,
        flag(f, "source-profile"),
        variant,
    )
    .map(|plan| plan.value())
    .map_err(bail)
}

pub fn cmd_source_plan(f: &Flags) -> ExitCode {
    let output = match derive_source_plan(f) {
        Ok(output) => tensorfs_core::canon::write(&output),
        Err(code) => return code,
    };
    if let Some(path) = flag(f, "out") {
        if let Err(error) = fs::write(path, &output) {
            eprintln!("REFUSED IO_FAILED: write source plan {path}: {error}");
            return ExitCode::FAILURE;
        }
    } else {
        println!("{}", String::from_utf8(output).unwrap());
    }
    ExitCode::SUCCESS
}

struct ClosedSourcePlan {
    converter: String,
    order: Vec<(String, String)>,
    registry: String,
    session: String,
    /// The component's CARRIER, member included. A closed plan re-prepares from these, so
    /// dropping the member here would read a store-resident sharded index as a safetensors
    /// file at exactly the point the plan was supposed to be replayable.
    sources: Vec<(String, CarrierInput)>,
    /// The plan's whole carrier set — the directory a `weight_map` resolves against.
    set: CarrierSet,
    target: String,
}

fn read_closed_source_plan(path: &Path) -> Result<ClosedSourcePlan, ExitCode> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!(
                "REFUSED IO_FAILED: read source plan {}: {error}",
                path.display()
            );
            return Err(ExitCode::FAILURE);
        }
    };
    let value = match canon::parse(&bytes, limits::DOC_MAX_BYTES) {
        Ok(value) => value,
        Err(error) => return Err(bail(error)),
    };
    if canon::write(&value) != bytes {
        eprintln!("REFUSED NON_CANONICAL: source plan bytes are not canonical JSON");
        return Err(ExitCode::FAILURE);
    }
    let mut fields = match Fields::new("SourcePlan", &value) {
        Ok(fields) => fields,
        Err(error) => return Err(bail(error)),
    };
    let carriers: Vec<CarrierInput> = match fields.req("carriers").and_then(|value| {
        as_arr("SourcePlan", "carriers", value).and_then(|values| {
            values
                .iter()
                .map(|value| {
                    let mut carrier = Fields::new("SourcePlanCarrier", value)?;
                    let path = PathBuf::from(carrier.req_str("path")?);
                    let member = match carrier.opt("member") {
                        Some(value) => {
                            Some(as_str("SourcePlanCarrier", "member", value)?.to_string())
                        }
                        None => None,
                    };
                    carrier.done()?;
                    Ok(CarrierInput { member, path })
                })
                .collect()
        })
    }) {
        Ok(carriers) => carriers,
        Err(error) => return Err(bail(error)),
    };
    let order = match fields.req("construction_order").and_then(|value| {
        as_arr("SourcePlan", "construction_order", value).and_then(|rows| {
            rows.iter()
                .map(|row| {
                    let values = as_arr("SourcePlan", "construction_order row", row)?;
                    if values.len() != 2 {
                        return tensorfs_core::err::refuse(
                            Code::WRONG_TYPE,
                            "source plan construction-order row must have two strings",
                        );
                    }
                    Ok((
                        as_str("SourcePlan", "component", &values[0])?.to_string(),
                        as_str("SourcePlan", "key", &values[1])?.to_string(),
                    ))
                })
                .collect()
        })
    }) {
        Ok(order) => order,
        Err(error) => return Err(bail(error)),
    };
    let converter = match fields.req_str("converter") {
        Ok(value) => value.to_string(),
        Err(error) => return Err(bail(error)),
    };
    let profile = match fields.req_str("profile") {
        Ok(value) => value.to_string(),
        Err(error) => return Err(bail(error)),
    };
    let registry = match fields.req_str("registry") {
        Ok(value) => value.to_string(),
        Err(error) => return Err(bail(error)),
    };
    let _registry_sha256 = match fields.req_str("registry_sha256") {
        Ok(value) => value,
        Err(error) => return Err(bail(error)),
    };
    let session = match fields.req_str("session") {
        Ok(value) => value.to_string(),
        Err(error) => return Err(bail(error)),
    };
    let sources = match fields.req("sources").and_then(|value| {
        as_arr("SourcePlan", "sources", value).and_then(|rows| {
            rows.iter()
                .map(|row| {
                    let mut source = Fields::new("SourcePlanSource", row)?;
                    let component = source.req_str("component")?.to_string();
                    let path = PathBuf::from(source.req_str("path")?);
                    let member = match source.opt("source_member") {
                        Some(value) => {
                            let member = as_str("SourcePlanSource", "source_member", value)?;
                            fingerprint::validate_source_member(member)?;
                            Some(member.to_string())
                        }
                        None => None,
                    };
                    match source.req("projected")? {
                        Value::Bool(_) => {}
                        other => {
                            return tensorfs_core::err::refuse(
                                Code::WRONG_TYPE,
                                format!(
                                    "SourcePlanSource.projected must be bool, got {}",
                                    other.kind()
                                ),
                            )
                        }
                    }
                    source.done()?;
                    Ok((component, CarrierInput { member, path }))
                })
                .collect()
        })
    }) {
        Ok(sources) => sources,
        Err(error) => return Err(bail(error)),
    };
    let target = match fields.req_str("target") {
        Ok(value) => value.to_string(),
        Err(error) => return Err(bail(error)),
    };
    if let Err(error) = fields.done() {
        return Err(bail(error));
    }

    // Re-derive from carrier headers, reviewed member labels, and the exact registry.
    // This is what prevents an edited plan from assigning equal headers by hand.
    let mut derive_flags = Vec::new();
    if registry != BUILTIN_REGISTRY {
        derive_flags.push(("registry".to_string(), registry.clone()));
    }
    derive_flags.push(("source-profile".to_string(), profile.clone()));
    for carrier in &carriers {
        let path = carrier.path.to_string_lossy();
        derive_flags.push((
            "carrier".to_string(),
            match &carrier.member {
                Some(member) => format!("{member}={path}"),
                None => path.to_string(),
            },
        ));
    }
    let derived = derive_source_plan(&derive_flags)?;
    if derived != value {
        eprintln!(
            "REFUSED IDENTITY_MISMATCH: source plan no longer equals the unique plan derived from its carrier headers and registry"
        );
        return Err(ExitCode::FAILURE);
    }
    Ok(ClosedSourcePlan {
        converter,
        order,
        registry,
        session,
        sources,
        set: CarrierSet::of(&carriers),
        target,
    })
}

fn prepare_closed_source_plan(
    path: &Path,
) -> Result<(ClosedSourcePlan, Prepared, Flags), ExitCode> {
    let closed = read_closed_source_plan(path)?;
    let registry_bytes = registry_bytes_at(&closed.registry)?;
    let registry = match FingerprintRegistry::parse(&registry_bytes) {
        Ok(registry) => registry,
        Err(error) => return Err(bail(error)),
    };
    let mut flags = Vec::new();
    if closed.registry != BUILTIN_REGISTRY {
        flags.push(("registry".to_string(), closed.registry.clone()));
    }
    let as_is = closed.converter == convert::IDENTITY;
    if as_is {
        flags.push(("source-profile".to_string(), source_core::AS_IS.to_string()));
    }
    for (component, source) in &closed.sources {
        flags.push((
            "source".to_string(),
            format!("{component}={}", source.path.to_string_lossy()),
        ));
    }
    // The plan was re-derived above, so its converter is the planner's: as-is has no registry.
    let mut prepared = source_core::prepare(
        &closed.target,
        &closed.sources,
        &closed.set,
        (!as_is).then_some(&registry),
        None,
    )
    .map_err(bail)?;
    if let Err(error) = prepared.plan.apply_order(&closed.order) {
        return Err(bail(error));
    }
    if prepared.plan.converter != closed.converter
        || source_core::session_of(&prepared) != closed.session
    {
        eprintln!(
            "REFUSED IDENTITY_MISMATCH: source plan converter/session differs from the existing prepare path"
        );
        return Err(ExitCode::FAILURE);
    }
    Ok((closed, prepared, flags))
}

pub fn cmd_plan_source(path: &Path) -> ExitCode {
    let (_, prepared, _) = match prepare_closed_source_plan(path) {
        Ok(result) => result,
        Err(code) => return code,
    };
    print_plan(&prepared);
    ExitCode::SUCCESS
}

/// `outer` carries the flags that describe THIS RUN rather than the plan: where the pod's
/// repo cache is mounted and how often the conversion should make itself resumable
/// elsewhere. They are deliberately not in the closed plan document, because a plan is a
/// claim about what will be produced and a mount point is a fact about one machine — the
/// same plan run with and without a cache must produce the same bytes.
pub fn cmd_run_source(root: &Path, path: &Path, outer: &Flags) -> ExitCode {
    let (closed, _, mut flags) = match prepare_closed_source_plan(path) {
        Ok(result) => result,
        Err(code) => return code,
    };
    for (key, value) in outer.iter() {
        if matches!(key.as_str(), "checkpoint-bytes") {
            flags.push((key.clone(), value.clone()));
        }
    }
    let order_path = root
        .join("tmp")
        .join(format!("source-plan-{}.order.json", closed.session));
    if let Err(error) = fs::create_dir_all(order_path.parent().unwrap()) {
        eprintln!("REFUSED IO_FAILED: create source-plan order directory: {error}");
        return ExitCode::FAILURE;
    }
    let order = Value::arr(
        closed
            .order
            .iter()
            .map(|(component, key)| {
                Value::arr(vec![Value::str(component.clone()), Value::str(key.clone())])
            })
            .collect(),
    );
    if let Err(error) = fs::write(&order_path, canon::write(&order)) {
        eprintln!("REFUSED IO_FAILED: write source-plan order: {error}");
        return ExitCode::FAILURE;
    }
    flags.push((
        "order".to_string(),
        order_path.to_string_lossy().to_string(),
    ));
    let result = cmd_run(root, &closed.target, &flags);
    let _ = fs::remove_file(order_path);
    result
}

// ------------------------------------------------------------------ plan

type Prepared = source_core::Prepared;

pub(crate) fn construction_order(f: &Flags) -> Result<Vec<(String, String)>, ExitCode> {
    let path = match flag(f, "order") {
        Some(path) => path,
        None => {
            eprintln!(
                "REFUSED CONSTRUCTION_ORDER_REQUIRED: artifact production requires \
                 --order <json>; TensorFS never infers execution order from names or carriers"
            );
            return Err(ExitCode::FAILURE);
        }
    };
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("REFUSED IO_FAILED: read construction order {path}: {error}");
            return Err(ExitCode::FAILURE);
        }
    };
    let value = match canon::parse(&bytes, limits::DOC_MAX_BYTES) {
        Ok(value) => value,
        Err(error) => return Err(bail(error)),
    };
    let rows = match tensorfs_core::canon::as_arr("construction order", "rows", &value) {
        Ok(rows) => rows,
        Err(error) => return Err(bail(error)),
    };
    let mut order = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let row = match tensorfs_core::canon::as_arr("construction order", "row", row) {
            Ok(row) if row.len() == 2 => row,
            Ok(row) => {
                eprintln!(
                    "REFUSED WRONG_TYPE: construction order row {index} has arity {}, expected 2",
                    row.len()
                );
                return Err(ExitCode::FAILURE);
            }
            Err(error) => return Err(bail(error)),
        };
        let component =
            match tensorfs_core::canon::as_str("construction order", "component", &row[0]) {
                Ok(value) => value.to_string(),
                Err(error) => return Err(bail(error)),
            };
        let key = match tensorfs_core::canon::as_str("construction order", "key", &row[1]) {
            Ok(value) => value.to_string(),
            Err(error) => return Err(bail(error)),
        };
        order.push((component, key));
    }
    Ok(order)
}

/// Headers in, plan out. The one function `plan`, `run` and the worker all go through, so
/// a preview and a run cannot disagree about what was going to happen.
/// `--source-profile as-is/1` stores the sources as they are, unclassified.
fn prepare(f: &Flags, target: &str) -> Result<Prepared, ExitCode> {
    let sources = sources(f)?;
    let registry = load_registry(f)?;
    let variant = flag(f, "spec-variant").and_then(|value| value.parse().ok());
    let as_is = flag(f, "source-profile") == Some(source_core::AS_IS);
    source_core::prepare(
        target,
        &unnamed(&sources),
        &CarrierSet::default(),
        (!as_is).then_some(&registry),
        variant,
    )
    .map_err(bail)
}

fn print_plan(p: &Prepared) {
    let pl = &p.plan;
    println!("converter    {} ({})", pl.converter, pl.class_name);
    if !p.conv.validated_on_real_bytes {
        println!("             ⚠ geometry banked from the v1 quarry, NOT yet validated on real bytes (tfs-010)");
    }
    println!(
        "ops          {} tensors — {} recontain, {} rekey, {} permute, {} unchanged",
        pl.ops.len(),
        pl.count(Effect::Recontain),
        pl.count(Effect::Rekey),
        pl.count(Effect::Permute),
        pl.count(Effect::None)
    );
    println!(
        "inherit      {} tensors carry by object reference (zero re-hash)",
        pl.inherited()
    );
    // The seam's own decomposition. proto-001 #322 measured the real one at 1,376,256 B
    // runs — 5,376x the synthetic fixture's 256 B — so the class printed here is the
    // difference between a copy that rides the staging pass and real budgeted work.
    if let Some((ops, bytes, runs, mean, class)) = pl.permutes() {
        println!(
            "permute      {ops} ops over {bytes} source B — {runs} runs, mean run {mean} B, \
             class {class}"
        );
    }
    println!(
        "dropped      {} marker tensors, {} role siblings folded into their base",
        pl.dropped_markers.len(),
        pl.folded_roles.len()
    );
    // Selection is PER TENSOR, so one component routinely cites several encodings. Printing
    // the count per SPEC DIGEST is the claim: an alias is a display name, never an identity.
    let encs = pl.encodings();
    println!("encodings    {} selected, per tensor:", encs.len());
    for (alias, id, roles, n) in &encs {
        println!(
            "  {n:>6}     {alias}  {}  roles {roles:?}",
            id.chars().skip(7).take(12).collect::<String>()
        );
    }
    match pl.declared_bytes() {
        Ok(n) => println!("declared     {n} B will be written"),
        Err(e) => println!("declared     REFUSED {}: {}", e.code.as_str(), e.detail),
    }
    println!(
        "tensor schema {}",
        tensor_schema_digest_of(&source_core::plan_tensor_schema(pl))
    );
    println!(
        "cost         {} B of headers read, {} tensor bytes read",
        pl.header_bytes,
        pl.tensor_bytes_read()
    );
}

pub fn cmd_plan(target: &str, f: &Flags) -> ExitCode {
    let t0 = Instant::now();
    let mut p = match prepare(f, target) {
        Ok(p) => p,
        Err(c) => return c,
    };
    if flag(f, "order").is_some() {
        let order = match construction_order(f) {
            Ok(order) => order,
            Err(code) => return code,
        };
        if let Err(error) = p.plan.apply_order(&order) {
            return bail(error);
        }
    } else {
        println!("order        UNBOUND preview only; run requires --order <json>");
    }
    for (i, b) in p.verdicts.iter().enumerate() {
        println!(
            "verdict      {} via {} -> {} ({} keys)",
            b.component, b.dialect, b.converter, p.fps[i].logical_keys
        );
    }
    print_plan(&p);
    // The canonical key set this plan produces, for cross-plane comparison against another
    // packaging's index. Emitted on request rather than always: it is 639 lines on the real
    // native DiT, and it exists so the convergence claim can be CHECKED by a caller that
    // holds the other packaging, without a second copy of the rekey map anywhere.
    if flag_on(f, "keys") {
        for o in &p.plan.ops {
            println!("key          {}", o.out_key);
        }
    }
    // The refusal a preview exists to deliver: a bit-identical copy must never stamp as a
    // new encoding, and finding that out costs a header read rather than a rented GPU.
    if let Err(e) = p.plan.check_converts_something() {
        eprintln!("REFUSED {}: {}", e.code.as_str(), e.detail);
        return ExitCode::FAILURE;
    }
    println!(
        "planned in {:.1} ms, peak RSS {:.1} MiB",
        t0.elapsed().as_secs_f64() * 1000.0,
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ the isolated run

/// What isolation this host could actually give the worker. Recorded in the receipt as an
/// integer, because claiming a tier you did not get is worse than getting a lower one.
///
/// The rlimits are RESOURCE fences — address space and file size bound a runaway, not
/// progress. There is deliberately no CPU-time limit: `ulimit -t` ended legitimate work at
/// a fixed number of CPU-seconds whatever the input (xs-007 row 4). Whether the worker is
/// still working is observed instead, by `watch::supervise`.
fn isolation(argv: &[String], f: &Flags) -> (Command, i64, String) {
    let me = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "tfs".to_string());
    let quoted: Vec<String> = argv
        .iter()
        .map(|a| format!("'{}'", a.replace('\'', "'\\''")))
        .collect();
    let as_kb: u64 = flag_num(f, "rlimit-as-mb", 2048) * 1024;
    let fsz_kb: u64 = flag_num(f, "rlimit-fsize-mb", 65536) * 1024;
    let inner = format!(
        "ulimit -v {as_kb}; ulimit -f {fsz_kb}; ulimit -c 0; exec '{me}' {}",
        quoted.join(" ")
    );

    // Namespace isolation when the host grants it; rlimits always. `unshare -rn` needs
    // unprivileged user namespaces, which this box does not grant — so the tier is
    // recorded rather than asserted.
    let netns = !flag_on(f, "no-netns")
        && Command::new("unshare")
            .args(["-rn", "/bin/true"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
    if netns {
        let mut c = Command::new("unshare");
        c.args(["-rn", "/bin/sh", "-c", &inner]);
        (
            c,
            2,
            format!("netns + rlimits (as {as_kb} KiB, fsize {fsz_kb} KiB)"),
        )
    } else {
        let mut c = Command::new("/bin/sh");
        c.args(["-c", &inner]);
        (
            c,
            1,
            format!(
                "separate process + rlimits (as {as_kb} KiB, fsize {fsz_kb} KiB); no netns on this host"
            ),
        )
    }
}

fn session_of(f: &Flags, p: &Prepared) -> String {
    flag(f, "session")
        .map(str::to_string)
        .unwrap_or_else(|| source_core::session_of(p))
}

/// The coordinator side of a run: it prepares and previews, then hands the whole job to an
/// isolated child. This process never opens the carrier's data region.
pub fn cmd_run(root: &Path, target: &str, f: &Flags) -> ExitCode {
    let mut p = match prepare(f, target) {
        Ok(p) => p,
        Err(c) => return c,
    };
    let order = match construction_order(f) {
        Ok(order) => order,
        Err(code) => return code,
    };
    if let Err(error) = p.plan.apply_order(&order) {
        return bail(error);
    }
    if let Err(e) = p.plan.check_converts_something() {
        eprintln!("REFUSED {}: {}", e.code.as_str(), e.detail);
        return ExitCode::FAILURE;
    }
    print_plan(&p);
    let session = session_of(f, &p);

    let mut argv: Vec<String> = vec![
        "ingest-worker".into(),
        root.display().to_string(),
        target.to_string(),
        "--session".into(),
        session.clone(),
    ];
    for (k, v) in f.iter() {
        if matches!(
            k.as_str(),
            "source"
                | "config"
                | "registry"
                | "spec-variant"
                | "order"
                | "tenant"
                | "isolation-tier"
                | "checkpoint-bytes"
                | "source-profile"
        ) {
            argv.push(format!("--{k}"));
            argv.push(v.clone());
        }
    }
    argv.push("--progress".into());
    let (mut cmd, tier, how) = isolation(&argv, f);
    println!("isolation    tier {tier} — {how}");
    println!("session      {session}");
    // The worker's stdout is the observation channel: every line it prints is relayed
    // here, and its progress records are consumed. Its stderr stays its own.
    cmd.stdout(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("REFUSED IO_FAILED: spawn worker: {e}");
            return ExitCode::FAILURE;
        }
    };
    // `sh` and `unshare` both exec in place, so this pid IS the worker's.
    println!("worker       pid {}", child.id());
    let stdout = child.stdout.take().expect("piped stdout");
    let declared = p.plan.declared_bytes().unwrap_or(0);
    let report = watch::supervise(&mut child, stdout, declared);
    let secs = report.elapsed.as_secs_f64();
    println!(
        "progress     moved {} B in {secs:.1} s ({:.1} MB/s), {} admitted; worst silence \
         {:.3} s, patience {:.1} s",
        report.moved,
        report.moved as f64 / secs.max(1e-9) / 1e6,
        report.admitted,
        report.worst_gap.as_secs_f64(),
        report.patience.as_secs_f64()
    );
    match report.verdict {
        Verdict::Exited(s) if s.success() => {
            println!("worker exited 0 in {secs:.1}s");
            ExitCode::SUCCESS
        }
        Verdict::Exited(s) => {
            eprintln!(
                "worker exited {:?} — the candidate stays reapable and nothing installed",
                s.code()
            );
            ExitCode::FAILURE
        }
        Verdict::Stalled { silent, patience } => {
            eprintln!(
                "worker stalled: no byte moved, no CPU consumed and never seen running for \
                 {:.1} s (patience {:.1} s); killed — the candidate stays reapable and \
                 nothing installed",
                silent.as_secs_f64(),
                patience.as_secs_f64()
            );
            ExitCode::FAILURE
        }
        Verdict::Runaway { silent, allowed } => {
            eprintln!(
                "worker runaway: burning CPU with no byte moved for {:.1} s, past the \
                 {:.1} s its own rate says the whole input costs; killed — the candidate \
                 stays reapable and nothing installed",
                silent.as_secs_f64(),
                allowed.as_secs_f64()
            );
            ExitCode::FAILURE
        }
        Verdict::Lost(e) => {
            eprintln!("REFUSED IO_FAILED: wait for worker: {e}");
            ExitCode::FAILURE
        }
    }
}

// ------------------------------------------------------------------ the hermetic worker

/// The worker's side of the observation channel: one record on stdout per chunk the store
/// moves and per object it admits, cumulative, so the parent reads totals and never has to
/// trust an interval. Under `--progress` only — a worker run by hand prints nothing extra.
#[derive(Default)]
struct Reporter {
    moved: AtomicU64,
    admitted: AtomicU64,
}

impl Reporter {
    fn announce(&self) {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let written = writeln!(
            out,
            "{PROGRESS_PREFIX}{}\t{}",
            self.moved.load(Ordering::Relaxed),
            self.admitted.load(Ordering::Relaxed)
        )
        .and_then(|()| out.flush());
        if written.is_err() {
            // The channel's other end is closed: the run that asked for these records is
            // gone. A worker nobody supervises does not go on burning CPU into a candidate
            // nobody will install; the session stays resumable for the next run.
            eprintln!(
                "worker: the supervising run is gone; stopping — the candidate stays resumable"
            );
            std::process::exit(1);
        }
    }
}

impl Progress for Reporter {
    fn moved(&self, bytes: u64) {
        self.moved.fetch_add(bytes, Ordering::Relaxed);
        self.announce();
    }
    fn admitted(&self) {
        self.admitted.fetch_add(1, Ordering::Relaxed);
        self.announce();
    }
}

pub fn cmd_worker(root: &Path, target: &str, f: &Flags) -> ExitCode {
    let t0 = Instant::now();
    let mut p = match prepare(f, target) {
        Ok(p) => p,
        Err(c) => return c,
    };
    let order = match construction_order(f) {
        Ok(order) => order,
        Err(code) => return code,
    };
    if let Err(error) = p.plan.apply_order(&order) {
        return bail(error);
    }
    if let Err(e) = p.plan.check_converts_something() {
        eprintln!("REFUSED {}: {}", e.code.as_str(), e.detail);
        return ExitCode::FAILURE;
    }
    let session = session_of(f, &p);
    let tenant = flag(f, "tenant").unwrap_or("dev").to_string();
    let mut store = ok!(Store::open(root));
    if flag_on(f, "progress") {
        store = store.observed(Arc::new(Reporter::default()));
    }

    // The root goes down FIRST. From this line on the session is reapable whatever happens.
    let (mut ingest_root, _operation_writer) = ok!(tx::open_root(&store, &session, &tenant));

    // Configs become canonical JSON bytes inside the CozyTensors header.
    let mut configs = Vec::new();
    for (k, v) in f.iter() {
        if k != "config" {
            continue;
        }
        let (name, path) = match v.split_once('=') {
            Some(x) => x,
            None => {
                eprintln!("REFUSED MISSING_FIELD: --config wants name=path");
                return ExitCode::FAILURE;
            }
        };
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("REFUSED IO_FAILED: read {path}: {e}");
                return ExitCode::FAILURE;
            }
        };
        configs.push((
            name.to_string(),
            ok!(tensorfs_core::header::canonical_config("config", &bytes)),
        ));
    }
    configs.sort_by(|a, b| a.0.cmp(&b.0));

    // The golden suite runs BEFORE any conversion: a converter whose own geometry does not
    // reassemble is not permitted to touch an artifact.
    let golden = ok!(source_core::golden_suite(p.conv));
    println!("golden       {golden} cases reassembled bit-for-bit");

    let predicted = tensor_schema_digest_of(&source_core::plan_tensor_schema(&p.plan));
    // Qualification fixtures are build data, not model-store objects.
    for (_, seed) in &p.seeds {
        if let Some(v) = &seed.vectors {
            ok!(v.check(&seed.spec));
        }
    }
    // THE DURABILITY HALF. The cache is the STORE's — `tfs store ensure --repo-cache` bound
    // it and this handle carries it — and it is a PATH and never a mode: an unbound Store is
    // a conversion with no cache, and a cache that is missing, read-only, full or unmounted
    // degrades to exactly that conversion. What it buys is that converted objects stop being
    // pod-local while the conversion is still running — the reason a released pod used to
    // cost the whole run, and the half the conversion journal beside it cannot supply
    // because it dies with the pod.
    let cache = store.repo_cache().is_some();
    let headers: Vec<(std::path::PathBuf, &SourceHeader)> = p
        .files
        .iter()
        .map(|file| (file.path.clone(), &file.header))
        .collect();
    // The SAME identity the conversion journal keys its file on. One plan-identity function,
    // so the two planes cannot disagree about which conversion this is.
    let plan_digest = format!(
        "sha256:{}",
        tensorfs_core::ingest::journal::plan_digest(&p.plan, &headers)
    );
    println!("plan         {plan_digest}");
    let policy = durability::Policy {
        chain: &session,
        operation: &session,
        plan: &plan_digest,
        interval: flag_num(f, "checkpoint-bytes", durability::INTERVAL_BYTES),
    };
    let (converted, durable) = durability::with(&store, &policy, |mirror| {
        tx::execute(
            &store,
            &p.plan,
            &p.files,
            &source_core::plain_specs(&p.seeds),
            &configs,
            &session,
            mirror,
        )
    });
    if cache {
        let mirrored = durable.mirror;
        println!(
            "write-through {} object(s) stored, {} already present, {} refused — {} B durable \
             over {} chain link(s){}",
            mirrored.stored,
            mirrored.present,
            mirrored.unavailable,
            mirrored.durable.bytes,
            durable.journal.links,
            if mirrored.durable.stalled {
                " (the cache stopped answering; this run is no longer resumable elsewhere)"
            } else {
                ""
            }
        );
        if let Some(head) = &durable.journal.head {
            // The chain is immutable and lives on the cache; THIS DIGEST IS THE MUTABLE
            // CELL, and it is printed rather than written because its home is the hub,
            // advanced by compare-and-set on the monotonic byte watermark beside it. A
            // mutable pointer on a volume shared by every pod of one owner is the hazard
            // the local-Store decision exists to refuse.
            println!(
                "chain        {} links {} bytes {}",
                head.id(),
                durable.journal.links,
                durable.journal.bytes
            );
        }
    }
    let mut out = ok!(converted);
    let got = out.header.tensor_schema_digest();
    if got != predicted {
        eprintln!(
            "REFUSED TENSOR_SCHEMA_MISMATCH: the plan predicted tensor schema {predicted} and \
             the written header projects {got} — the preview and the run disagree"
        );
        return ExitCode::FAILURE;
    }

    // The profile the reviewed hit authorized, made explicit and stored as evidence.
    let recipe_bytes = canon::write(&Value::obj(vec![
        ("class", Value::str(p.plan.class_name.clone())),
        ("converter", Value::str(p.plan.converter.clone())),
        (
            // Each component's ALLOWED SET, one entry per component. A component may be
            // named more than once in a target (that is how a mixed-allocation carrier is
            // addressed), and a map keyed on the component with a bare alias value emits
            // the same key twice — a document the canonical reader refuses on the way back.
            "target",
            Value::map({
                let mut out: Vec<(String, Value)> = Vec::new();
                for (c, _) in &p.target.components {
                    if out.iter().any(|(k, _)| k == c) {
                        continue;
                    }
                    let aliases = match p.target.allowed(c) {
                        Ok(v) => v,
                        Err(e) => return bail(e),
                    };
                    out.push((
                        c.clone(),
                        Value::arr(aliases.into_iter().map(Value::str).collect()),
                    ));
                }
                out
            }),
        ),
    ]));
    let recipe = ok!(store.put_stream_held(
        &mut recipe_bytes.as_slice(),
        Some(&ObjectRef::of(&recipe_bytes)),
        &Default::default(),
        Some(&session),
    ))
    .obj;
    let build = tensorfs_core::ids::object_id(
        format!("{}|{}", p.conv.name, env!("CARGO_PKG_VERSION")).as_bytes(),
    );
    let source_bytes = match p.files.iter().try_fold(0u64, |total, source| {
        total.checked_add(source.header.file_len)
    }) {
        Some(bytes) => bytes,
        None => {
            return bail(tensorfs_core::err::Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "conversion input byte count overflow".into(),
            })
        }
    };
    let logical_output_bytes = ok!(p.plan.declared_bytes());
    let logical_output_observation = match i64::try_from(logical_output_bytes) {
        Ok(bytes) => bytes,
        Err(_) => {
            return bail(tensorfs_core::err::Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "conversion output exceeds signed accounting extent".into(),
            })
        }
    };
    let profile = IngestProfile {
        class: "normal".to_string(),
        source_dialect: p.conv.dialect.to_string(),
        converter_build: build.clone(),
        converter_recipe: recipe.clone(),
        expected_tensor_schema_digest: predicted.clone(),
        allowed_encodings: {
            let mut v: Vec<ObjectRef> = p.seeds.iter().map(|(_, s)| s.spec.object_ref()).collect();
            v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
            v.dedup_by(|a, b| a.sha256 == b.sha256);
            v
        },
        // The receipt quota counts materialized bytes. A reviewed permutation can
        // duplicate a low-rank factor into several canonical projections; its exact
        // planned output, not only the carrier length, must fit the same bound.
        max_source_bytes: source_bytes.max(logical_output_bytes),
        max_tensors: limits::MAX_TENSORS as u64,
    };
    ok!(profile.validate(&registry::platform_digests()));
    let profile_ref = profile.object_ref();

    let subject = IngestSubject {
        profile: profile_ref,
        tenant: tenant.clone(),
        sources: p
            .files
            .iter()
            .zip(p.raws.iter())
            .map(|(sf, raw)| (source_core::source_digest(&sf.header, raw), 1u64))
            .collect(),
        proposed_header: out.header_ref.clone(),
        // The candidate set is EVERY AND ONLY what the manifest reaches. Stamp re-walks it
        // and refuses on any difference, so a hand-written short list would simply fail.
        candidates: ok!(ck::walk(&store, &out.manifest))
            .distinct()
            .into_iter()
            .cloned()
            .collect(),
    };
    let subject_ref = subject.object_ref();

    let isolation_tier: u64 = flag_num(f, "isolation-tier", 1);
    out.obs
        .push(("source_bytes".to_string(), profile.max_source_bytes as i64));
    out.obs.push((
        "logical_output_bytes".to_string(),
        logical_output_observation,
    ));
    out.obs.push(("golden_cases".to_string(), golden as i64));
    out.obs
        .push(("isolation_tier".to_string(), isolation_tier as i64));
    out.obs
        .push(("peak_rss_kb".to_string(), rss_kb("VmHWM") as i64));
    out.obs
        .push(("elapsed_ms".to_string(), t0.elapsed().as_millis() as i64));
    out.obs.push((
        "converter_validated_on_real_bytes".to_string(),
        i64::from(p.conv.validated_on_real_bytes),
    ));
    out.obs.sort();

    // THE STAMP. Nothing here builds a receipt by hand any more: the receipt is what
    // `stamp` returns after re-deriving every fact from bytes the store holds, so a run
    // that reported success but wrote something else mints nothing.
    // `golden_suite` REFUSES on any reassembly mismatch, so reaching here means it ran
    // clean. The case COUNT rides the accounting; the verdict is what it is. `absent` has a
    // real producer: a VALUE-MOVING converter with no banked fixtures never ran a suite at
    // all, and a suite that did not run is not a suite that passed. An identity converter
    // moves no value and has nothing to reassemble, so zero cases is its complete suite.
    let golden_verdict = if golden > 0 || p.conv.class == Class::Identity {
        stamp::GOLDEN_PASS
    } else {
        stamp::GOLDEN_ABSENT
    };
    let stamped = ok!(stamp::stamp(
        &store,
        stamp::Presented {
            subject: &subject,
            profile: &profile,
            profile_ref: &subject.profile.clone(),
            outcome: &out,
            golden: golden_verdict,
            isolation_tier,
            evidence: Evidence::ApprovedProducer {
                implementation: build,
                recipe,
            },
            signer: &format!("tfs-ingest-worker/{}", env!("CARGO_PKG_VERSION")),
            signature: "unsigned:dev",
            pin: &registry::platform_digests(),
        }
    ));
    let stamp_ref = stamped.stamp.object_ref();
    let receipt = stamped.receipt;
    let receipt_ref = receipt.object_ref();

    ingest_root.candidates = vec![
        out.header_ref.clone(),
        out.manifest_ref.clone(),
        subject_ref,
        stamp_ref.clone(),
        receipt_ref.clone(),
    ];
    ok!(tx::write_session(store.root(), &ingest_root));
    let dir = tx::candidate_dir(store.root(), &session);
    for (name, bytes) in [
        ("receipt.json", receipt.canonical_bytes()),
        ("stamp.json", stamped.stamp.canonical_bytes()),
        ("subject.json", subject.canonical_bytes()),
        ("profile.json", profile.canonical_bytes()),
    ] {
        if let Err(e) = fs::write(dir.join(name), bytes) {
            eprintln!("REFUSED IO_FAILED: write {name}: {e}");
            return ExitCode::FAILURE;
        }
    }

    for (k, v) in &receipt.observations {
        println!("  {k:<36} {v}");
    }
    println!("candidate    header {}", out.header_ref.id());
    println!("candidate    manifest {}", out.manifest_ref.id());
    println!("stamp        {}", stamp_ref.id());
    println!("  tensor schema {}", stamped.stamp.tensor_schema_digest);
    println!("  candidates {}", stamped.stamp.candidate_set);
    println!("  golden     {}", stamped.stamp.golden);
    println!("receipt      {}", receipt_ref.id());
    println!(
        "candidate    {} is resumable until explicit reap — NOT a checkpoint until `tfs ingest install`",
        session
    );
    ExitCode::SUCCESS
}

/// Bytes for a golden case.
///
/// This uses the keyed splitmix stream rather than an arithmetic ramp, and the reason is a
/// defect this arm actually caught: with `src[i] = 31i + 7 (mod 256)` the pattern's period
/// is 256, which DIVIDES both the interleaved stride (768 B) and the flat stride (256 B).
/// Under that pattern the correct head-interleaved split and the wrong flat split produce
/// BYTE-IDENTICAL output, and the wrong-order arm silently cannot fire. A test pattern
/// whose period shares a factor with the geometry under test proves nothing.
pub fn cmd_reingest(root: &Path, header_hex: &str, target: &str, f: &Flags) -> ExitCode {
    let t0 = Instant::now();
    let store = ok!(Store::open(root));
    let hlen = match fs::metadata(store.object_path(header_hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {header_hex}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let href = ObjectRef {
        sha256: header_hex.to_string(),
        length: hlen,
    };
    let header = ok!(ck::load_header(&store, &href));
    let target = ok!(Target::parse(target));
    let variant = flag(f, "spec-variant").and_then(|value| value.parse().ok());
    let seeds = match source_core::specs_for(&target, variant) {
        Ok(s) => s,
        Err(error) => return bail(error),
    };
    let mut rekey = Rekey::default();
    for (k, v) in f.iter() {
        if k == "rekey" {
            rekey.rules.push(ok!(Rekey::parse(v)));
        }
    }

    let views: Vec<Source<'_>> = header
        .components
        .iter()
        .map(|(c, _)| Source::Canonical {
            component: c.clone(),
            header: &header,
        })
        .collect();
    let conv = ok!(convert::converter("cozytensors.inherit/1"));
    let mut plan = ok!(convert::plan(
        conv,
        &views,
        &target,
        &source_core::plain_specs(&seeds),
        &rekey
    ));
    drop(views);
    let order = match construction_order(f) {
        Ok(order) => order,
        Err(code) => return code,
    };
    ok!(plan.apply_order(&order));

    println!("converter    {} ({})", plan.converter, plan.class_name);
    println!(
        "ops          {} tensors — {} rekey, {} unchanged; {} inherit by object reference",
        plan.ops.len(),
        plan.count(Effect::Rekey),
        plan.count(Effect::None),
        plan.inherited()
    );
    println!(
        "objects      {} distinct segment objects carry WITHOUT being read",
        convert::inherited_objects(&plan).len()
    );
    if let Err(e) = plan.check_converts_something() {
        eprintln!("REFUSED {}: {}", e.code.as_str(), e.detail);
        return ExitCode::FAILURE;
    }

    let session = flag(f, "session").unwrap_or("reingest").to_string();
    let tenant = flag(f, "tenant").unwrap_or("dev").to_string();
    let (mut ingest_root, _operation_writer) = ok!(tx::open_root(&store, &session, &tenant));
    // A reingest re-derives an artifact already in this Store from objects already in this
    // Store. It moves nothing a fresh pod would have to re-download, so it has nothing to
    // write through.
    let out = ok!(tx::execute(
        &store,
        &plan,
        &[],
        &source_core::plain_specs(&seeds),
        &header.configs,
        &session,
        None,
    ));
    ingest_root.candidates = vec![out.header_ref.clone(), out.manifest_ref.clone()];
    ok!(tx::write_session(store.root(), &ingest_root));

    for k in [
        "bytes_hashed",
        "bytes_written",
        "bytes_inherited",
        "tensors",
        "parts",
        "segments",
    ] {
        println!("  {k:<20} {}", out.get(k));
    }
    println!(
        "CAS-asserted: {} B inherited, {} B hashed — {}",
        out.get("bytes_inherited"),
        out.get("bytes_hashed"),
        if out.get("bytes_hashed") == 0 {
            "ZERO re-hash, ZERO re-upload"
        } else {
            "SOMETHING WAS RE-HASHED"
        }
    );
    println!("candidate    header {}", out.header_ref.id());
    println!("candidate    manifest {}", out.manifest_ref.id());
    println!(
        "in {:.1} ms, peak RSS {:.1} MiB",
        t0.elapsed().as_secs_f64() * 1000.0,
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ install / reap

/// The coordinator's act. Nothing else turns a candidate session into a repository release.
pub fn cmd_install(
    root: &Path,
    session: &str,
    org: &str,
    name: &str,
    version: &str,
    lane: &str,
    flags: &Flags,
) -> ExitCode {
    let store = ok!(Store::open(root));
    let dir = tx::candidate_dir(store.root(), session);
    let read = |n: &str| fs::read(dir.join(n));
    let (rb, sb, tb, pb) = match (
        read("receipt.json"),
        read("subject.json"),
        read("stamp.json"),
        read("profile.json"),
    ) {
        (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
        _ => {
            eprintln!(
                "REFUSED ROOT_ABSENT: no candidate {session:?} — reaped, never created, or \
                 already installed"
            );
            return ExitCode::FAILURE;
        }
    };
    let receipt = ok!(IngestVerificationReceipt::parse(&rb));
    let subject = ok!(IngestSubject::parse(&sb));
    let stamp_doc = ok!(Stamp::parse(&tb));
    let _profile = ok!(IngestProfile::parse(&pb));
    let snap = ok!(ck::load_manifest(&store, &receipt.manifest));
    let header = ok!(ck::load_header(&store, &receipt.header));
    let repo = ok!(tensorfs_core::repository::RepositoryName::new(org, name));
    let _operation_writer = ok!(tx::resume_operation(
        &store,
        session,
        &receipt.manifest,
        &snap,
    ));
    let id = ok!(tx::install(
        &store,
        tx::Install {
            session,
            receipt: &receipt,
            subject: &subject,
            manifest: &snap,
            header: &header,
            stamp: &stamp_doc,
            repo,
            version,
            lane,
            observed_repository: flag(flags, "observed"),
        },
    ));
    println!("installed    {id} as {org}/{name} {version} {lane}");
    println!("             from receipt {}", receipt.object_ref().id());
    ExitCode::SUCCESS
}

pub fn cmd_reap(root: &Path, _f: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let (gone, kept) = ok!(tx::reap(&store));
    for g in &gone {
        println!("reaped       {g}");
    }
    println!(
        "{} reaped, {} live, {} released manifests (unchanged by reaping)",
        gone.len(),
        kept.len(),
        tx::installed(&store).len()
    );
    ExitCode::SUCCESS
}

pub fn cmd_sessions(root: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let installed = tx::installed(&store);
    for i in &installed {
        println!("manifest     sha256:{i}");
    }
    let live = tx::sessions(&store);
    for s in &live {
        println!("candidate    {s} (temporary — not a release)");
    }
    println!(
        "{} released manifests, {} candidate sessions",
        installed.len(),
        live.len()
    );
    ExitCode::SUCCESS
}

/// Run every reviewed converter's geometry live, and plant the wrong one beside it.
///
/// The RED arm here is the defect the v1 quarry pinned numerically: a flat three-way split
/// of a head-interleaved fused QKV. It produces the right KEY, the right DTYPE and the right
/// SHAPE, and the wrong numbers — the quarry measured max|d| 6.09e+01 against |ref|max
/// 6.78e+01, roughly 90% error, and nothing raised. Here the two outputs are the same length
/// and a different digest, which is the same fact stated in bytes.
pub fn cmd_golden() -> ExitCode {
    let (mut pass, mut red, mut failed) = (0usize, 0usize, 0usize);
    for c in convert::CONVERTERS.iter() {
        for (name, x) in convert::golden_cases(c) {
            let src = source_core::golden_bytes(name, x.in_bytes());
            match x.reassembles(&src) {
                Ok(true) => {
                    pass += 1;
                    println!(
                        "  ok   {:<50} {} B -> {} B",
                        name,
                        x.in_bytes(),
                        x.out_bytes()
                    );
                }
                Ok(false) => {
                    failed += 1;
                    println!("  FAIL {name}: does not reassemble its own input");
                }
                Err(e) => {
                    failed += 1;
                    println!("  FAIL {name}: {}", e.code.as_str());
                }
            }

            if let convert::Xform::SwapHalves { .. } = &x {
                let mut swapped = Vec::new();
                ok!(x.apply(&src, &mut swapped));
                if swapped.len() == src.len() && swapped != src {
                    red += 1;
                    println!(
                        "  RED  {name}: unchanged gate/value order differs from converted bytes"
                    );
                } else {
                    failed += 1;
                    println!("  MISS {name}: unchanged fused halves were accepted");
                }
                continue;
            }

            // The planted wrong geometry: the SAME source read as one contiguous group.
            {
                let convert::Xform::QkvSplit {
                    groups,
                    shares,
                    take,
                    unit_bytes,
                } = &x
                else {
                    unreachable!("half-swap handled above")
                };
                if *groups == 1 {
                    continue;
                }
                let flat = convert::Xform::QkvSplit {
                    groups: 1,
                    shares: *shares,
                    take: *take,
                    unit_bytes: unit_bytes * groups,
                };
                let (mut a, mut b) = (Vec::new(), Vec::new());
                ok!(x.apply(&src, &mut a));
                ok!(flat.apply(&src, &mut b));
                if a.len() == b.len() && a != b {
                    red += 1;
                    println!(
                        "  RED  {:<50} flat split: same {} B, digest {} vs {}",
                        format!("{name} read as ONE group"),
                        a.len(),
                        &tensorfs_core::sha256::hex_digest(&a)[..12],
                        &tensorfs_core::sha256::hex_digest(&b)[..12]
                    );
                } else {
                    failed += 1;
                    println!("  MISS {name}: the wrong split produced the same bytes");
                }
            }
        }
    }
    println!("  {pass} golden cases reassembled, {red} wrong-order arms fired");
    if failed == 0 && red > 0 && pass > 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// What the converters are and what evidence stands behind each.
pub fn cmd_converters() -> ExitCode {
    for c in convert::CONVERTERS.iter() {
        println!(
            "{:<24} {:<12} dialect {:<26} {}",
            c.name,
            c.class.name(),
            c.dialect,
            if c.validated_on_real_bytes {
                "validated on real bytes"
            } else {
                "QUARRY-DERIVED, awaiting real bytes (tfs-010)"
            }
        );
        println!("  {}", c.doc);
        for s in convert::seams(c) {
            println!(
                "  seam   {:<34} head dim from {}, members {:?}",
                s.fused, s.norm, s.members
            );
        }
        for (name, x) in convert::golden_cases(c) {
            println!(
                "  golden {name:<34} {} B in, {} B out",
                x.in_bytes(),
                x.out_bytes()
            );
        }
    }
    let _ = (Dtype::U8, Class::Identity);
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ restamp

/// Re-run `Stamp` over a candidate session's documents, against the bytes the store holds.
///
/// Two jobs. It is the RE-VERIFICATION surface a coordinator has (th-002 verifies a receipt
/// against exactly these facts and must never build a second checker), and it is where the
/// harness plants defects: `--plant <what>` corrupts ONE presented fact and the run shows
/// which refusal that fact is protected by. The plants live in this binary, beside tfs-002's
/// `--fault` kill points and never in the library — a library that can be told to lie is a
/// library nobody can trust in production.
pub fn cmd_restamp(root: &Path, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let dir = tx::candidate_dir(store.root(), session);
    let read = |n: &str| match fs::read(dir.join(n)) {
        Ok(b) => Ok(b),
        Err(_) => {
            eprintln!("REFUSED ROOT_ABSENT: no candidate {session:?} document {n:?}");
            Err(ExitCode::FAILURE)
        }
    };
    let (pb, sb, rb) = match (
        read("profile.json"),
        read("subject.json"),
        read("receipt.json"),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => return ExitCode::FAILURE,
    };
    let mut profile = ok!(IngestProfile::parse(&pb));
    let mut subject = ok!(IngestSubject::parse(&sb));
    let receipt = ok!(IngestVerificationReceipt::parse(&rb));

    // The outcome, reconstructed from the store rather than remembered: the header and
    // manifest are objects, and the accounting rode the receipt.
    let header = ok!(ck::load_header(&store, &subject.proposed_header));
    let closure = ok!(ck::load_closure(&store, &header));
    let manifest = ok!(ck::load_manifest(&store, &receipt.manifest));
    let mut out = tx::Outcome {
        header_ref: subject.proposed_header.clone(),
        manifest_ref: receipt.manifest.clone(),
        header,
        closure,
        manifest,
        obs: receipt.observations.clone(),
    };
    let mut evidence = receipt.evidence.clone();
    let mut golden = flag(flags, "golden")
        .unwrap_or(stamp::GOLDEN_PASS)
        .to_string();

    let plant = flag(flags, "plant").unwrap_or("");
    match plant {
        "" => {}
        "tensor-schema" => {
            profile.expected_tensor_schema_digest = tensorfs_core::ids::object_id(b"other")
        }
        "candidate" => {
            subject.candidates.pop();
        }
        "candidate-surplus" => subject
            .candidates
            .push(ObjectRef::of(b"an object nothing reaches")),
        "encoding" => profile.allowed_encodings.clear(),
        "quota-tensors" => profile.max_tensors = 1,
        "quota-bytes" => profile.max_source_bytes = 1,
        "converter" => {
            evidence = Evidence::ApprovedProducer {
                implementation: profile.converter_build.clone(),
                recipe: ObjectRef::of(b"a recipe this store never admitted"),
            }
        }
        "evidence-metadata" => {
            // The header-invisible claim: a source's own `__metadata__` saying "quantized by
            // X" is restated as an evidence arm. It names a digest of the CLAIM, and the
            // store holds no such object, so it resolves to nothing.
            evidence = Evidence::IndependentNumerical {
                verifier_build: tensorfs_core::ids::object_id(b"verifier"),
                report: ObjectRef::of(br#"{"quantized_by":"trust me","method":"fp8"}"#),
            }
        }
        "golden-absent" => golden = stamp::GOLDEN_ABSENT.to_string(),
        // The profile is DIGEST-BOUND to its subject, so altering it alone is caught before
        // any of its contents are read. This plant proves that binding on its own.
        "profile-bytes" => profile.max_tensors += 1,
        "structure" => {
            // One tensor's logical shape moved: the header no longer projects the tensor schema
            // the profile pinned, and the manifest no longer describes what is in the store.
            if let Some((_, ts)) = out.header.components.first_mut() {
                if let Some((_, t)) = ts.first_mut() {
                    t.shape.push(1);
                }
            }
        }
        o => {
            eprintln!("REFUSED UNKNOWN_FIELD: no plant {o:?}");
            return ExitCode::FAILURE;
        }
    }
    if !plant.is_empty() {
        println!("plant        {plant}");
    }
    // A plant that alters the PROFILE re-points the subject at it, so the run exercises the
    // check that fact is protected by rather than stopping at the digest binding. The
    // `profile-bytes` plant is the one that deliberately does not, and proves the binding.
    if matches!(
        plant,
        "tensor-schema" | "encoding" | "quota-tensors" | "quota-bytes"
    ) {
        subject.profile = profile.object_ref();
    }
    let stamped = ok!(stamp::stamp(
        &store,
        stamp::Presented {
            profile_ref: &subject.profile.clone(),
            subject: &subject,
            profile: &profile,
            outcome: &out,
            golden: &golden,
            isolation_tier: 1,
            evidence,
            signer: &receipt.signer,
            signature: &receipt.signature,
            pin: &registry::platform_digests(),
        }
    ));
    println!("restamped    {}", stamped.stamp.object_id());
    println!("  tensor schema {}", stamped.stamp.tensor_schema_digest);
    println!("  candidates {}", stamped.stamp.candidate_set);
    println!("receipt      {}", stamped.receipt.object_id());
    if stamped.receipt.object_id() == receipt.object_id() {
        println!("             IDENTICAL to the receipt the border minted");
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- the gguf-v1 planner

/// `tfs gguf plan <file>` — derive what a serving lane would take, from the metadata
/// region alone. `tfs gguf admit <file>` runs the same derivation and then asks the plan
/// for its admission verdict, which today refuses for every real artifact.
///
/// The report is deterministic and carries no path: the same bytes under any name produce
/// the same `plan` digest, which is the filename-Q-label arm made checkable rather than
/// argued.
pub fn cmd_gguf_plan(path: &Path, admit: bool, flags: &Flags) -> ExitCode {
    let plan = ok!(tensorfs_core::ingest::gguf_plan(
        path,
        flag_on(flags, "probe")
    ));
    let mut r = String::new();
    let hist = plan.type_histogram();
    let (mut reversed, mut from_kv) = (0u64, 0u64);
    for t in &plan.tensors {
        match t.shape_source {
            tensorfs_core::ingest::ShapeSource::DimsReversed => reversed += 1,
            tensorfs_core::ingest::ShapeSource::OrigShapeKv => from_kv += 1,
        }
    }
    r += &format!(
        "plan         GGUF v{}, {} tensors, alignment {}\n",
        plan.version,
        plan.tensors.len(),
        plan.alignment
    );
    r += &format!(
        "architecture {:?} — DECLARED by the producer, never admission evidence\n",
        plan.architecture
    );
    r += &format!(
        "labels       file_type {} — recorded, never consulted; the histogram below is \
         measured off the tensor info table\n",
        plan.file_type_label.clone().unwrap_or("(absent)".into())
    );
    r += &format!(
        "metadata     {} B, {} KV of which {} orig_shape\n",
        plan.metadata_bytes, plan.kv_count, plan.orig_shape_kvs
    );
    r += &format!(
        "carrier blob {} B whole, opaque, streamed by a path-consuming engine{}\n",
        plan.carrier_blob_bytes(),
        match plan.probe {
            true => " (DERIVED: the input is a bounded metadata probe, bytes not present)",
            false => " (present on disk)",
        }
    );
    r += &format!(
        "decomposed   {} objects over a {} B data region at offset {}\n",
        plan.tensors.len(),
        plan.data_bytes,
        plan.data_offset
    );
    r += &format!(
        "shapes       {reversed} from reversed dims, {from_kv} from the producer's \
         orig_shape KV\n"
    );
    for (name, n, bytes) in &hist {
        let spec = plan
            .tensors
            .iter()
            .find(|t| &t.type_name == name)
            .and_then(|t| t.plain_dtype)
            .map(|d| format!("plain/1 {d}"))
            .unwrap_or_else(|| "UNREVIEWED — no EncodingSpec describes it".into());
        r += &format!("  {name:<8} {n:>5} tensors {bytes:>15} B  {spec}\n");
    }
    let show: usize = flag(flags, "tensors")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for t in plan.tensors.iter().take(show) {
        r += &format!(
            "  {:<52} {:<7} carrier {:?} -> logical {:?} {} blocks @{} +{}\n",
            t.name, t.type_name, t.carrier_dims, t.logical_shape, t.blocks, t.offset, t.nbytes
        );
    }
    print!("{r}");
    println!(
        "plan digest  {}",
        tensorfs_core::ids::object_id(r.as_bytes())
    );
    if let Some(twin) = flag(flags, "twin") {
        let bytes = match fs::read(twin) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("REFUSED IO_FAILED: read {twin}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let n = ok!(tensorfs_core::ingest::gguf_check_twin(&plan, &bytes));
        println!("twin         {n} logical shapes agree with {twin}");
    }
    match plan.admission() {
        Ok(()) => {
            println!("admission    every tensor is described by a reviewed spec");
            ExitCode::SUCCESS
        }
        Err(e) if admit => bail(e),
        Err(e) => {
            println!("admission    NOT OPEN — {e}");
            ExitCode::SUCCESS
        }
    }
}
