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
//!                        an expiring quarantine root, and emits a receipt. It cannot
//!                        install a snapshot and it has no network stack linked into it.
//!   `install`            the COORDINATOR's act. It consumes the receipt, verifies the
//!                        binding, walks the tree, and alone promotes one root.
//!   `reap`               what happens to everything that never reached `install`.
//!
//! A candidate is never a checkpoint. That is the whole design, and `roots/` is the only
//! place the difference is visible.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::checkpoint as ck;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{topology_digest_of, topology_value};
use tensorfs_core::ids::{ConfigRef, Doc, ObjectRef};
use tensorfs_core::ingest::carrier::{self, SourceHeader};
use tensorfs_core::ingest::convert::{self, Class, Converter, Effect, Plan, Rekey, Source, Target};
use tensorfs_core::ingest::fingerprint::{self, Banked, Fingerprint, FingerprintRegistry};
use tensorfs_core::ingest::stamp::{self, Stamp};
use tensorfs_core::ingest::transaction::{self as tx, SourceFile};
use tensorfs_core::ingest::{Evidence, IngestProfile, IngestSubject, IngestVerificationReceipt};
use tensorfs_core::limits;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::Store;
use tensorfs_core::structure::{Provenance, REAL};

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

/// A carrier's exact source identity, without reading its 5 GB of body.
///
/// The header pins every key, dtype, shape and offset in the file, so a digest over the
/// header bytes plus the file length names the exact generation. Hashing the whole artifact
/// would be a stronger claim and a pointless one: the border is about to hash every byte it
/// actually admits, and the ones it does not admit are not part of this identity.
fn source_digest(h: &SourceHeader, raw_header: &[u8]) -> String {
    tensorfs_core::ids::object_id(&canon::write(&Value::obj(vec![
        (
            "carrier_header_sha256",
            Value::str(tensorfs_core::sha256::hex_digest(raw_header)),
        ),
        ("data_start", Value::uint(h.data_start)),
        ("file_len", Value::uint(h.file_len)),
        ("tensors", Value::uint(h.tensors.len() as u64)),
    ])))
}

/// A committed header JSON — REAL evidence with no artifact behind it (the shipped H3 DiT
/// inventory is 66 GB; its header is 486 KiB and is what the repo actually has). It goes
/// through the exact same strict decoder, with the data region reconstructed from the
/// declared geometry, so the evidence is decoded by the code that decodes artifacts.
fn read_header_only(p: &Path) -> tensorfs_core::err::Result<(SourceHeader, Vec<u8>)> {
    let json = fs::read(p).map_err(|e| tensorfs_core::err::Refusal {
        code: tensorfs_core::err::Code::IO_FAILED,
        detail: format!("read {}: {e}", p.display()),
    })?;
    let data_start = json.len() as u64 + 8;
    let h = carrier::parse_header(&json, data_start, None)?;
    let mut raw = (json.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(&json);
    Ok((h, raw))
}

/// Read a carrier header AND keep its exact bytes, for the identity above. A
/// `*.index.json` is a SHARDED carrier (the official H3 trees ship 13/14-shard
/// components) and decodes through `read_sharded` — every single-file law per shard,
/// exact weight_map cover both directions, virtual joined offsets. Any other `.json` is
/// committed header evidence.
fn read_carrier(p: &Path) -> tensorfs_core::err::Result<(SourceHeader, Vec<u8>)> {
    if p.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.ends_with(".index.json"))
        .unwrap_or(false)
    {
        return carrier::read_sharded(p);
    }
    if p.extension().map(|e| e == "json").unwrap_or(false) {
        return read_header_only(p);
    }
    let h = carrier::read_header(p)?;
    let mut f = fs::File::open(p).map_err(|e| tensorfs_core::err::Refusal {
        code: tensorfs_core::err::Code::IO_FAILED,
        detail: format!("reopen {}: {e}", p.display()),
    })?;
    let mut raw = vec![0u8; h.header_bytes as usize + 8];
    f.read_exact(&mut raw)
        .map_err(|e| tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("reread header of {}: {e}", p.display()),
        })?;
    Ok((h, raw))
}

fn registry_path(f: &Flags) -> PathBuf {
    PathBuf::from(flag(f, "registry").unwrap_or("vectors/fingerprints/registry.json"))
}

fn load_registry(f: &Flags) -> Result<FingerprintRegistry, ExitCode> {
    let p = registry_path(f);
    match fs::read(&p) {
        Ok(b) => match FingerprintRegistry::parse(&b) {
            Ok(r) => Ok(r),
            Err(e) => {
                eprintln!("REFUSED {}: {}: {}", e.code.as_str(), p.display(), e.detail);
                Err(ExitCode::FAILURE)
            }
        },
        // An ABSENT registry is not an empty one: an empty registry would authorize by
        // silence. It refuses, and says how to fill it.
        Err(_) => {
            eprintln!(
                "REFUSED UNREGISTERED_FINGERPRINT: no reviewed registry at {} — \
                 bank one with `tfs ingest bank` before any carrier can be authorized",
                p.display()
            );
            Err(ExitCode::FAILURE)
        }
    }
}

/// alias -> the CANDIDATE SET it names. An alias that groups several physical variants (the
/// two `fp8-rowwise/1` scale geometries, the two `fp8-scaled-scalar/1` role sets one real
/// artifact carries) contributes all of them: the alias is no longer ambiguous at the
/// component, because selection happens per TENSOR from the siblings the carrier supplies.
/// `--spec-variant` still pins ONE, which is exactly the old single-spec path.
fn specs_of(
    alias: &str,
    variant: Option<usize>,
) -> tensorfs_core::err::Result<Vec<registry::Seed>> {
    let hits: Vec<registry::Seed> = registry::seeds()
        .into_iter()
        .filter(|s| s.alias == alias)
        .collect();
    match (hits.len(), variant) {
        (0, _) => tensorfs_core::err::refuse(
            tensorfs_core::err::Code::UNKNOWN_ENCODING,
            format!("{alias:?} is not a platform alias"),
        ),
        (n, Some(i)) if i < n => Ok(vec![hits.into_iter().nth(i).unwrap()]),
        (n, Some(i)) => tensorfs_core::err::refuse(
            tensorfs_core::err::Code::UNKNOWN_ENCODING,
            format!("{alias:?} groups {n} specs; --spec-variant {i} names none of them"),
        ),
        (_, None) => Ok(hits),
    }
}

fn specs_for(t: &Target, f: &Flags) -> Result<Vec<(String, registry::Seed)>, ExitCode> {
    let variant: Option<usize> = flag(f, "spec-variant").and_then(|s| s.parse().ok());
    // plain/1 is always resolved: a real component is MIXED, and the tensors its producer's
    // eligibility predicate skipped are plain storage beside the encoded ones.
    let mut out: Vec<(String, registry::Seed)> = match specs_of("plain/1", None) {
        Ok(s) => s.into_iter().map(|s| ("plain/1".to_string(), s)).collect(),
        Err(e) => return Err(bail(e)),
    };
    for (_, alias) in &t.components {
        if out.iter().any(|(a, _)| a == alias) {
            continue;
        }
        match specs_of(alias, variant) {
            Ok(v) => out.extend(v.into_iter().map(|s| (alias.clone(), s))),
            Err(e) => return Err(bail(e)),
        }
    }
    Ok(out)
}

/// The (alias, spec) view the planner and executor take.
fn plain_specs(seeds: &[(String, registry::Seed)]) -> Vec<(String, EncodingSpec)> {
    seeds
        .iter()
        .map(|(a, s)| (a.clone(), s.spec.clone()))
        .collect()
}

/// The projected topology of a PLAN — computable before a byte moves, which is what lets a
/// profile pin the exact result the conversion must produce.
fn plan_topology(p: &Plan) -> Value {
    topology_value(p.ops.iter().map(|o| {
        (
            o.component.as_str(),
            o.out_key.as_str(),
            o.logical_dtype,
            o.logical_shape.as_slice(),
        )
    }))
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
        let (h, raw) = ok!(read_carrier(path));
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
        println!("  topology     {}", fp.topology_digest);
        println!("  source       {}", source_digest(&h, &raw));
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
                    "  verdict      AUTHORIZED {}/{} via {} -> {}",
                    b.task_family, b.structural_variant, b.dialect, b.converter
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
pub fn cmd_bank(
    dialect: &str,
    family: &str,
    variant: &str,
    converter: &str,
    f: &Flags,
) -> ExitCode {
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
    let p = registry_path(f);
    let mut reg = match fs::read(&p) {
        Ok(b) => ok!(FingerprintRegistry::parse(&b)),
        Err(_) => FingerprintRegistry::default(),
    };
    for (component, path) in &srcs {
        let (h, raw) = ok!(read_carrier(path));
        let fp = ok!(fingerprint::fingerprint(component, &h));
        let sd = source_digest(&h, &raw);
        reg.merge(Banked {
            keyset_digest: fp.keyset_digest.clone(),
            dialect: dialect.to_string(),
            task_family: family.to_string(),
            structural_variant: variant.to_string(),
            component: component.clone(),
            converter: converter.to_string(),
            topology_digest: fp.topology_digest.clone(),
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
            fp.keyset_digest, fp.topology_digest
        );
    }
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    // The REVIEW ACT does not get to leave behind a document the border cannot read. A
    // `--source-name` with one em dash in it wrote a registry that every later run refused
    // NON_ASCII_FIELD, at the far end of the pipeline and with the review already "done".
    let bytes = reg.canonical_bytes();
    ok!(FingerprintRegistry::parse(&bytes));
    if let Err(e) = fs::write(&p, &bytes) {
        eprintln!("REFUSED IO_FAILED: write {}: {e}", p.display());
        return ExitCode::FAILURE;
    }
    println!("{} entries -> {}", reg.entries.len(), p.display());
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ plan

struct Prepared {
    plan: Plan,
    conv: &'static Converter,
    seeds: Vec<(String, registry::Seed)>,
    files: Vec<SourceFile>,
    raws: Vec<Vec<u8>>,
    verdicts: Vec<Banked>,
    fps: Vec<Fingerprint>,
    target: Target,
}

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
fn prepare(f: &Flags, target: &str) -> Result<Prepared, ExitCode> {
    let srcs = sources(f)?;
    let reg = load_registry(f)?;
    let target = match Target::parse(target) {
        Ok(t) => t,
        Err(e) => return Err(bail(e)),
    };
    let seeds = specs_for(&target, f)?;
    let specs = plain_specs(&seeds);

    let (mut files, mut raws, mut verdicts, mut fps) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (component, path) in &srcs {
        let (h, raw) = match read_carrier(path) {
            Ok(v) => v,
            Err(e) => return Err(bail(e)),
        };
        let fp = match fingerprint::fingerprint(component, &h) {
            Ok(v) => v,
            Err(e) => return Err(bail(e)),
        };
        // A REVIEWED registry hit IS the profile authorization. Ambiguity and absence both
        // refuse; neither falls back to a filename, a directory, or a config.
        let banked = match reg.authorize(component, &fp) {
            Ok(b) => b,
            Err(e) => return Err(bail(e)),
        };
        verdicts.push(banked);
        fps.push(fp);
        raws.push(raw);
        files.push(SourceFile {
            path: path.clone(),
            header: h,
        });
    }

    // One artifact, one converter. Two components whose reviewed entries disagree about
    // the dialect is a review defect, not a merge.
    let names: Vec<&str> = verdicts.iter().map(|b| b.converter.as_str()).collect();
    if names.windows(2).any(|w| w[0] != w[1]) {
        eprintln!(
            "REFUSED AMBIGUOUS_CLASSIFICATION: the components' reviewed entries name \
             different converters {names:?}"
        );
        return Err(ExitCode::FAILURE);
    }
    let conv = match convert::converter(names[0]) {
        Ok(c) => c,
        Err(e) => return Err(bail(e)),
    };

    let views: Vec<Source<'_>> = files
        .iter()
        .enumerate()
        .map(|(i, sf)| Source::Carrier {
            component: srcs[i].0.clone(),
            file: i,
            header: &sf.header,
        })
        .collect();
    let plan = match convert::plan(conv, &views, &target, &specs, &Rekey::default()) {
        Ok(p) => p,
        Err(e) => return Err(bail(e)),
    };
    drop(views);
    Ok(Prepared {
        plan,
        conv,
        seeds,
        files,
        raws,
        verdicts,
        fps,
        target,
    })
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
    println!("topology     {}", topology_digest_of(&plan_topology(pl)));
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
            "verdict      {} -> {}/{} via {} ({} keys)",
            b.component, b.task_family, b.structural_variant, b.dialect, p.fps[i].logical_keys
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
fn isolation(argv: &[String], f: &Flags) -> (Command, i64, String) {
    let me = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "tfs".to_string());
    let quoted: Vec<String> = argv
        .iter()
        .map(|a| format!("'{}'", a.replace('\'', "'\\''")))
        .collect();
    let as_kb: u64 = flag_num(f, "rlimit-as-mb", 2048) * 1024;
    let cpu_s: u64 = flag_num(f, "rlimit-cpu-s", 3600);
    let fsz_kb: u64 = flag_num(f, "rlimit-fsize-mb", 65536) * 1024;
    let inner = format!(
        "ulimit -v {as_kb}; ulimit -t {cpu_s}; ulimit -f {fsz_kb}; ulimit -c 0; exec '{me}' {}",
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
            format!("netns + rlimits (as {as_kb} KiB, cpu {cpu_s} s)"),
        )
    } else {
        let mut c = Command::new("/bin/sh");
        c.args(["-c", &inner]);
        (
            c,
            1,
            format!(
                "separate process + rlimits (as {as_kb} KiB, cpu {cpu_s} s); no netns on this host"
            ),
        )
    }
}

fn session_of(f: &Flags, p: &Prepared) -> String {
    if let Some(s) = flag(f, "session") {
        return s.to_string();
    }
    let v = Value::obj(vec![
        ("converter", Value::str(p.plan.converter.clone())),
        (
            "sources",
            Value::arr(
                p.files
                    .iter()
                    .zip(p.raws.iter())
                    .map(|(sf, raw)| Value::str(source_digest(&sf.header, raw)))
                    .collect(),
            ),
        ),
        (
            "topology",
            Value::str(topology_digest_of(&plan_topology(&p.plan))),
        ),
        (
            "order",
            Value::arr(
                p.plan
                    .ops
                    .iter()
                    .map(|op| {
                        Value::arr(vec![
                            Value::str(op.component.clone()),
                            Value::str(op.out_key.clone()),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    tensorfs_core::ids::object_id(&canon::write(&v))
        .trim_start_matches("sha256:")
        .chars()
        .take(16)
        .collect()
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
                | "ttl"
                | "isolation-tier"
        ) {
            argv.push(format!("--{k}"));
            argv.push(v.clone());
        }
    }
    let (mut cmd, tier, how) = isolation(&argv, f);
    println!("isolation    tier {tier} — {how}");
    println!("session      {session}");
    let t0 = Instant::now();
    match cmd.status() {
        Ok(s) if s.success() => {
            println!("worker exited 0 in {:.1}s", t0.elapsed().as_secs_f64());
            ExitCode::SUCCESS
        }
        Ok(s) => {
            eprintln!(
                "worker exited {:?} — the quarantine root stays reapable and nothing installed",
                s.code()
            );
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("REFUSED IO_FAILED: spawn worker: {e}");
            ExitCode::FAILURE
        }
    }
}

// ------------------------------------------------------------------ the hermetic worker

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
    let ttl: u64 = flag_num(f, "ttl", limits::INGEST_ROOT_TTL_SECS);
    let store = ok!(Store::init(root));

    // The root goes down FIRST. From this line on the session is reapable whatever happens.
    let mut ingest_root = ok!(tx::open_root(&store, &session, &tenant, ttl));

    // Configs are sidecar bytes the header commits BY REFERENCE.
    let mut configs: Vec<(String, ConfigRef)> = Vec::new();
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
        let put = ok!(store.put_stream(
            &mut bytes.as_slice(),
            Some(&ObjectRef::of(&bytes)),
            &Default::default()
        ));
        configs.push((
            name.to_string(),
            ConfigRef {
                sha256: put.obj.sha256,
                length: put.obj.length,
                media_type: "application/json".to_string(),
            },
        ));
    }
    configs.sort_by(|a, b| a.0.cmp(&b.0));

    // The golden suite runs BEFORE any conversion: a converter whose own geometry does not
    // reassemble is not permitted to touch an artifact.
    let golden = ok!(golden_suite(p.conv));
    println!("golden       {golden} cases reassembled bit-for-bit");

    let predicted = topology_digest_of(&plan_topology(&p.plan));
    // A spec's `vectors` reference must resolve INSIDE the closure it ships with, so the
    // vector documents are admitted before the header that will name them.
    for (_, seed) in &p.seeds {
        if let Some(v) = &seed.vectors {
            ok!(ck::put_doc(&store, v));
        }
    }
    let out = ok!(tx::execute(
        &store,
        &p.plan,
        &p.files,
        &plain_specs(&p.seeds),
        &configs
    ));
    let got = out.header.topology_digest();
    if got != predicted {
        eprintln!(
            "REFUSED TOPOLOGY_MISMATCH: the plan predicted {predicted} and the written \
             header projects {got} — the preview and the run disagree"
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
    let recipe = ok!(store.put_stream(
        &mut recipe_bytes.as_slice(),
        Some(&ObjectRef::of(&recipe_bytes)),
        &Default::default()
    ))
    .obj;
    let build = tensorfs_core::ids::object_id(
        format!("{}|{}", p.conv.name, env!("CARGO_PKG_VERSION")).as_bytes(),
    );
    let profile = IngestProfile {
        class: "normal".to_string(),
        source_dialect: p.conv.dialect.to_string(),
        converter_build: build.clone(),
        converter_recipe: recipe.clone(),
        expected_topology_digest: predicted.clone(),
        allowed_encodings: {
            let mut v: Vec<ObjectRef> = p.seeds.iter().map(|(_, s)| s.spec.object_ref()).collect();
            v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
            v.dedup_by(|a, b| a.sha256 == b.sha256);
            v
        },
        max_source_bytes: p.files.iter().map(|s| s.header.file_len).sum(),
        max_tensors: limits::MAX_TENSORS as u64,
    };
    ok!(profile.validate(&registry::platform_digests()));
    let profile_ref = ok!(ck::put_doc(&store, &profile));

    let subject = IngestSubject {
        profile: profile_ref,
        tenant: tenant.clone(),
        sources: p
            .files
            .iter()
            .zip(p.raws.iter())
            .map(|(sf, raw)| (source_digest(&sf.header, raw), 1u64))
            .collect(),
        proposed_header: out.header_ref.clone(),
        // The candidate set is EVERY AND ONLY what the snapshot reaches. Stamp re-walks it
        // and refuses on any difference, so a hand-written short list would simply fail.
        candidates: ok!(ck::walk(&store, &out.snapshot))
            .distinct()
            .into_iter()
            .cloned()
            .collect(),
    };
    let subject_ref = ok!(ck::put_doc(&store, &subject));

    let isolation_tier: u64 = flag_num(f, "isolation-tier", 1);
    let mut out = out;
    out.obs
        .push(("source_bytes".to_string(), profile.max_source_bytes as i64));
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
    let stamp_ref = ok!(ck::put_doc(&store, &stamped.stamp));
    let receipt = stamped.receipt;
    let receipt_ref = ok!(ck::put_doc(&store, &receipt));

    ingest_root.candidates = vec![
        out.header_ref.clone(),
        out.snapshot_ref.clone(),
        subject_ref,
        stamp_ref.clone(),
        receipt_ref.clone(),
    ];
    ok!(tx::write_root(store.root(), &ingest_root));
    let dir = tx::quarantine_dir(store.root(), &session);
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
    println!("candidate    snapshot {}", out.snapshot_ref.id());
    println!("stamp        {}", stamp_ref.id());
    println!("  topology   {}", stamped.stamp.topology_digest);
    println!("  candidates {}", stamped.stamp.candidate_set);
    println!("  golden     {}", stamped.stamp.golden);
    println!("receipt      {}", receipt_ref.id());
    println!(
        "quarantine   {} expires at {} — NOT a checkpoint until `tfs ingest install`",
        session, ingest_root.expires_at
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
fn golden_bytes(seed: &str, n: u64) -> Vec<u8> {
    use std::io::Read;
    let mut v = vec![0u8; n as usize];
    crate::ckpt::Noise::new(seed, n)
        .read_exact(&mut v)
        .expect("keyed stream");
    v
}

/// Every reviewed converter's geometry, exercised on real byte patterns before it is
/// allowed near an artifact. A transform that does not reassemble its own input is the
/// head-interleave-vs-concat error, which does not crash and serves garbage.
fn golden_suite(c: &Converter) -> tensorfs_core::err::Result<usize> {
    let cases = convert::golden_cases(c);
    let mut n = 0;
    for (name, x) in cases {
        let src = golden_bytes(name, x.in_bytes());
        if !x.reassembles(&src)? {
            return tensorfs_core::err::refuse(
                tensorfs_core::err::Code::GOLDEN_MISMATCH,
                format!(
                    "{}: golden case {name} does not reassemble its own input",
                    c.name
                ),
            );
        }
        n += 1;
    }
    Ok(n)
}

// ------------------------------------------------------------------ re-ingest

/// Re-ingest an ALREADY-CANONICAL checkpoint. This is the inherit-by-reference proof:
/// nothing is read, nothing is hashed, nothing is uploaded — the segment references simply
/// carry into a new header, and the CAS is asserted to already hold every one of them.
///
/// With no `--rekey`, the plan changes nothing at all and CONVERTS_NOTHING fires: a
/// bit-identical copy must never stamp as a new encoding, and the existing checkpoint
/// already IS the result. With one, the keys move and the bytes do not.
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
    let seeds = match specs_for(&target, f) {
        Ok(s) => s,
        Err(c) => return c,
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
        &plain_specs(&seeds),
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
    let mut ingest_root = ok!(tx::open_root(
        &store,
        &session,
        &tenant,
        flag_num(f, "ttl", limits::INGEST_ROOT_TTL_SECS)
    ));
    let out = ok!(tx::execute(
        &store,
        &plan,
        &[],
        &plain_specs(&seeds),
        &header.configs
    ));
    ingest_root.candidates = vec![out.header_ref.clone(), out.snapshot_ref.clone()];
    ok!(tx::write_root(store.root(), &ingest_root));

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
    println!("candidate    snapshot {}", out.snapshot_ref.id());
    println!(
        "in {:.1} ms, peak RSS {:.1} MiB",
        t0.elapsed().as_secs_f64() * 1000.0,
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ install / reap

/// The coordinator's act. Nothing else in this binary writes `roots/`.
pub fn cmd_install(root: &Path, session: &str) -> ExitCode {
    let store = ok!(Store::open(root));
    let dir = tx::quarantine_dir(store.root(), session);
    let read = |n: &str| fs::read(dir.join(n));
    let (rb, sb, tb) = match (
        read("receipt.json"),
        read("subject.json"),
        read("stamp.json"),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
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
    let snap = ok!(ck::load_manifest(&store, &receipt.snapshot));
    let header = ok!(ck::load_header(&store, &receipt.header));
    let id = ok!(tx::install(
        &store, session, &receipt, &subject, &snap, &header, &stamp_doc
    ));
    println!("installed    {id}");
    println!("             from receipt {}", receipt.object_ref().id());
    ExitCode::SUCCESS
}

pub fn cmd_reap(root: &Path, f: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    // `--now` reaps every root regardless of age: the operator asserting the timeout.
    let now = if flag_on(f, "now") {
        u64::MAX
    } else {
        tx::now_secs()
    };
    let (gone, kept) = ok!(tx::reap(&store, now));
    for g in &gone {
        println!("reaped       {g}");
    }
    println!(
        "{} reaped, {} live, {} installed checkpoints (unchanged by reaping)",
        gone.len(),
        kept.len(),
        tx::installed(&store).len()
    );
    ExitCode::SUCCESS
}

pub fn cmd_roots(root: &Path) -> ExitCode {
    let store = ok!(Store::open(root));
    let installed = tx::installed(&store);
    for i in &installed {
        println!("installed    sha256:{i}");
    }
    let (_, live) = ok!(tx::reap(&store, 0));
    for s in &live {
        println!("candidate    {s} (quarantined — not a checkpoint)");
    }
    println!("{} installed, {} candidates", installed.len(), live.len());
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
    let (mut pass, mut red) = (0usize, 0usize);
    for c in convert::CONVERTERS.iter() {
        for (name, x) in convert::golden_cases(c) {
            let src = golden_bytes(name, x.in_bytes());
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
                Ok(false) => println!("  FAIL {name}: does not reassemble its own input"),
                Err(e) => println!("  FAIL {name}: {}", e.code.as_str()),
            }

            // The planted wrong geometry: the SAME source read as one contiguous group.
            {
                let convert::Xform::QkvSplit {
                    groups,
                    shares,
                    take,
                    unit_bytes,
                } = &x;
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
                    println!("  MISS {name}: the wrong split produced the same bytes");
                }
            }
        }
    }
    println!("  {pass} golden cases reassembled, {red} wrong-order arms fired");
    if red > 0 && pass > 0 {
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

/// Re-run `Stamp` over a quarantine session's documents, against the bytes the store holds.
///
/// Two jobs. It is the RE-VERIFICATION surface a coordinator has (th-002 verifies a receipt
/// against exactly these facts and must never build a second checker), and it is where the
/// harness plants defects: `--plant <what>` corrupts ONE presented fact and the run shows
/// which refusal that fact is protected by. The plants live in this binary, beside tfs-002's
/// `--fault` kill points and never in the library — a library that can be told to lie is a
/// library nobody can trust in production.
pub fn cmd_restamp(root: &Path, session: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let dir = tx::quarantine_dir(store.root(), session);
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
    // snapshot are objects, and the accounting rode the receipt.
    let header = ok!(ck::load_header(&store, &subject.proposed_header));
    let closure = ok!(ck::load_closure(&store, &header));
    let snapshot = ok!(ck::load_manifest(&store, &receipt.snapshot));
    let mut out = tx::Outcome {
        header_ref: subject.proposed_header.clone(),
        snapshot_ref: receipt.snapshot.clone(),
        header,
        closure,
        snapshot,
        obs: receipt.observations.clone(),
    };
    let mut evidence = receipt.evidence.clone();
    let mut golden = flag(flags, "golden")
        .unwrap_or(stamp::GOLDEN_PASS)
        .to_string();

    let plant = flag(flags, "plant").unwrap_or("");
    match plant {
        "" => {}
        "topology" => profile.expected_topology_digest = tensorfs_core::ids::object_id(b"other"),
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
            // One tensor's logical shape moved: the header no longer projects the topology
            // the profile pinned, and the snapshot no longer describes what is in the store.
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
        "topology" | "encoding" | "quota-tensors" | "quota-bytes"
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
    println!("  topology   {}", stamped.stamp.topology_digest);
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
