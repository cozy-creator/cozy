//! The layout plane's CLI (tfs-009): `check_compatibility`, hub grading, the round-trip
//! registration act, and the hermetic verifier's bounded document interface.
//!
//! This file is a CONSUMER of the two decision modules. It prints grades because someone
//! has to; it never decides one.

use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint as ck;
use tensorfs_core::compat::{
    self, Accepts, CodeTopology, CompatibilityReport, Lane, Requirement, Satisfaction,
};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::grade::{self, ConversionRegistration, Grade, GradeDecision, Mapping, Registry};
use tensorfs_core::header::Header;
use tensorfs_core::ids::{ObjectRef, Plain};
use tensorfs_core::ingest::fingerprint::MARKER_SUFFIXES;
use tensorfs_core::limits;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::Store;
use tensorfs_core::verifier::{self, VerifyRequest};

use crate::{bail, flag, flag_on, rss_kb, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

fn markers() -> Vec<String> {
    MARKER_SUFFIXES.iter().map(|s| s.to_string()).collect()
}

fn read(p: &Path) -> Result<Vec<u8>, ExitCode> {
    fs::read(p).map_err(|e| {
        eprintln!("REFUSED IO_FAILED: read {}: {e}", p.display());
        ExitCode::FAILURE
    })
}

macro_rules! rd {
    ($p:expr) => {
        match read($p) {
            Ok(b) => b,
            Err(c) => return c,
        }
    };
}

/// A header's own topology, projected into a code topology. This is the IDENTITY contract:
/// what a `load()` that constructs exactly this checkpoint would require. It is the
/// starting point for every real pair — you edit one side and grade the difference.
pub fn cmd_topology(root: &Path, hex: &str, flags: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let length = match fs::metadata(store.object_path(hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {hex}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let header = ok!(ck::load_header(
        &store,
        &ObjectRef {
            sha256: hex.to_string(),
            length,
        }
    ));
    if let Some(p) = flag(flags, "emit-header") {
        let _ = fs::write(p, ok!(header.canonical_bytes()));
    }
    let code = project(&header, flag(flags, "contract").unwrap_or("local"));
    emit(&code, flags);
    println!("contract     {}", code.contract);
    println!("tensors      {}", code.count());
    println!("topology     {}", code.topology_digest());
    ExitCode::SUCCESS
}

pub fn project(h: &Header, contract_name: &str) -> CodeTopology {
    CodeTopology {
        contract: tensorfs_core::ids::object_id(contract_name.as_bytes()),
        components: h
            .components
            .iter()
            .map(|(c, ts)| {
                (
                    c.clone(),
                    ts.iter()
                        .map(|(k, t)| {
                            (
                                k.clone(),
                                Requirement {
                                    dtype: t.dtype,
                                    shape: t.shape.clone(),
                                    accepts: Accepts::OneOf(vec![t.encoding.clone()]),
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
    }
}

/// Build a canonical header + code topology out of NEUTRAL measured rows.
///
/// The rows carry component, key, logical dtype/shape and the measured role geometry, and
/// nothing else — no format marks, no canonical bytes. That split is deliberate: the
/// transcriber that reads a pinned foreign header must never become
/// a second canonical writer, and the one-codec fence would catch it if it tried.
///
/// **The logical facts are REAL and the part object ids are PLACEHOLDERS.** Keys, dtypes,
/// shapes and role geometry come from upstream header bytes pinned by digest; the weights
/// themselves are hundreds of GB and never transit this machine. Nothing below the topology
/// layer is read by `check_compatibility` — that is what the containment ladder means — so
/// the placeholders are provably not load-bearing, and the provenance line says so out loud
/// rather than letting a reader assume bytes were verified.
pub fn cmd_import(rows_p: &Path, flags: &Flags) -> ExitCode {
    let bytes = rd!(rows_p);
    let v = ok!(tensorfs_core::canon::parse(&bytes, limits::DOC_MAX_BYTES));
    let mut f = match tensorfs_core::canon::Fields::new("rows", &v) {
        Ok(f) => f,
        Err(e) => return bail(e),
    };
    let component = ok!(f.req_str("component")).to_string();
    let provenance = ok!(f.req_str("provenance")).to_string();
    let tensors = ok!(tensorfs_core::canon::as_arr(
        "rows",
        "tensors",
        ok!(f.req("tensors"))
    ));

    let seeds = registry::seeds();

    // `--survey` reports EVERY distinct measured role set and the spec each one selects,
    // instead of stopping at the first the registry does not describe. An artifact can mix
    // encodings per tensor (tfs-003 f474111), so "the first gap" is not the answer to "what
    // is in this file" — and a survey is an observation, never an admission: it writes no
    // document and mints nothing.
    let survey = flag(flags, "survey").is_some();
    let mut seen: std::collections::BTreeMap<String, (Option<String>, u64, String)> =
        std::collections::BTreeMap::new();

    let mut closure = tensorfs_core::header::Closure::default();
    let mut out: Vec<(String, tensorfs_core::header::Tensor)> = Vec::new();
    let mut reqs: Vec<(String, Requirement)> = Vec::new();
    for tv in tensors {
        let mut tf = match tensorfs_core::canon::Fields::new("row", tv) {
            Ok(x) => x,
            Err(e) => return bail(e),
        };
        let dtype = ok!(Dtype::parse(ok!(tf.req_str("dtype"))));
        let key = ok!(tf.req_str("key")).to_string();
        let mut roles: Vec<(String, Dtype, Vec<u64>)> = Vec::new();
        if let Some(rv) = tf.opt("roles") {
            for r in ok!(tensorfs_core::canon::as_arr("row", "roles", rv)) {
                let mut rf = match tensorfs_core::canon::Fields::new("role", r) {
                    Ok(x) => x,
                    Err(e) => return bail(e),
                };
                let rdt = ok!(Dtype::parse(ok!(rf.req_str("dtype"))));
                let rname = ok!(rf.req_str("role")).to_string();
                let rshape = ok!(uints(ok!(rf.req("shape"))));
                roles.push((rname, rdt, rshape));
            }
        }
        let shape = ok!(uints(ok!(tf.req("shape"))));
        roles.sort_by(|a, b| a.0.cmp(&b.0));

        let measured = match roles.is_empty() {
            true => format!("plain {}", dtype.name()),
            false => roles
                .iter()
                .map(|(n, d, _)| format!("{n}:{}", d.name()))
                .collect::<Vec<_>>()
                .join(" "),
        };
        let spec = match roles.is_empty() {
            true => seeds
                .iter()
                .map(|s| &s.spec)
                .find(|s| s.roles.len() == 1 && s.roles[0].0 == "value")
                .cloned()
                .expect("plain/1 is entry zero of the platform namespace"),
            false => match pick(&seeds, dtype, &shape, &roles) {
                Some(s) => s,
                None => {
                    let detail =
                        format!(
                        "REFUSED UNKNOWN_ENCODING: {key}: no platform spec has roles {:?} with \
                         these carriers — the measured geometry is real and the registry does \
                         not describe it, which is a registry gap, never a reason to round it \
                         to a spec whose byte layout is a different one",
                        roles.iter().map(|(n, d, s)| format!("{n}:{}/{:?}", d.name(), s))
                            .collect::<Vec<_>>()
                    );
                    if !survey {
                        eprintln!("{detail}");
                        return ExitCode::FAILURE;
                    }
                    let e = seen.entry(measured).or_insert((None, 0, key.clone()));
                    e.1 += 1;
                    continue;
                }
            },
        };
        if survey {
            let alias = seeds
                .iter()
                .find(|s| s.spec.object_id() == spec.object_id())
                .map(|s| s.alias)
                .unwrap_or("?");
            let e = seen.entry(measured).or_insert((
                Some(format!("{alias} {}", &spec.object_id()[7..19])),
                0,
                key.clone(),
            ));
            e.1 += 1;
        }
        let sid = spec.object_id();
        if closure.get(&sid).is_none() {
            closure.insert(spec.clone());
        }
        let mut parts = Vec::new();
        for (name, r) in &spec.roles {
            let elements = ok!(tensorfs_core::dtype::checked_elements(&key, &shape));
            let pshape = ok!(r.shape.eval(&shape, elements));
            let carrier = if roles.is_empty() {
                dtype
            } else {
                roles
                    .iter()
                    .find(|(n, _, _)| n == name)
                    .map(|(_, d, _)| *d)
                    .unwrap_or(dtype)
            };
            let nbytes = ok!(tensorfs_core::dtype::checked_bytes(&key, &pshape, carrier));
            parts.push((
                name.clone(),
                placeholder_part(&key, name, carrier, pshape, nbytes),
            ));
        }
        out.push((
            key.clone(),
            tensorfs_core::header::Tensor {
                dtype,
                shape: shape.clone(),
                encoding: sid.clone(),
                parts,
            },
        ));
        reqs.push((
            key,
            Requirement {
                dtype,
                shape,
                accepts: Accepts::OneOf(vec![sid]),
            },
        ));
    }
    if survey {
        println!("survey       {provenance}");
        let (mut ok, mut gap) = (0u64, 0u64);
        for (measured, (spec, n, first)) in &seen {
            match spec {
                Some(s) => {
                    ok += n;
                    println!("  {n:5}  {measured:<52} {s}");
                }
                None => {
                    gap += n;
                    println!("  {n:5}  {measured:<52} GAP — {first}");
                }
            }
        }
        println!(
            "  {ok} tensors described, {gap} in {} registry gap(s)",
            seen.values().filter(|(s, _, _)| s.is_none()).count()
        );
        return ExitCode::SUCCESS;
    }
    let mut encodings: Vec<EncodingSpec> =
        closure.specs.iter().map(|(_, spec)| spec.clone()).collect();
    encodings.sort_by_key(EncodingSpec::object_id);
    let header = tensorfs_core::header::Header {
        configs: vec![],
        assets: vec![],
        encodings,
        components: vec![(component.clone(), out)],
    };
    ok!(header.validate(&closure));
    let code = CodeTopology {
        contract: tensorfs_core::ids::object_id(provenance.as_bytes()),
        components: vec![(component, reqs)],
    };
    if let Some(p) = flag(flags, "out-header") {
        let _ = fs::write(p, ok!(header.canonical_bytes()));
    }
    if let Some(p) = flag(flags, "out-code") {
        let _ = fs::write(p, code.canonical_bytes());
    }
    if let Some(d) = flag(flags, "out-closure") {
        let _ = fs::create_dir_all(d);
        for (id, s) in &closure.specs {
            let _ = fs::write(
                Path::new(d).join(format!("{}.json", &id[7..19])),
                s.canonical_bytes(),
            );
        }
    }
    println!(
        "imported     {} tensors from {provenance}",
        header.tensors().count()
    );
    println!("  logical    REAL — keys/dtypes/shapes/roles off pinned upstream header bytes");
    println!("  parts      PLACEHOLDER object ids: no weight bytes transit this machine");
    println!("header       {}", ok!(header.object_id()));
    println!("topology     {}", header.topology_digest());
    println!("code         {}", code.object_id());
    for (id, s) in &closure.specs {
        println!(
            "  spec       {} roles {:?}",
            &id[7..19],
            s.roles.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
    }
    ExitCode::SUCCESS
}

/// Select the platform spec whose EXACT role set, carriers and evaluated shapes reproduce
/// the geometry that was measured. Nothing is inferred from the logical dtype: this is the
/// whole `fp8 implies rank-2` trap, and the only escape from it is to match on what the
/// bytes actually declare. No hit is a registry gap, reported as one.
fn pick(
    seeds: &[registry::Seed],
    logical: Dtype,
    shape: &[u64],
    roles: &[(String, Dtype, Vec<u64>)],
) -> Option<EncodingSpec> {
    let elements = tensorfs_core::dtype::checked_elements("pick", shape).ok()?;
    seeds
        .iter()
        .map(|s| &s.spec)
        .find(|s| {
            if !s.admits_logical(logical) || s.roles.len() != roles.len() {
                return false;
            }
            if let Some(r) = s.logical_rank {
                if r as usize != shape.len() {
                    return false;
                }
            }
            s.roles
                .iter()
                .zip(roles.iter())
                .all(|((n, r), (rn, rd, rs))| {
                    n == rn
                        && r.carrier.admits(logical, *rd)
                        && r.shape
                            .eval(shape, elements)
                            .map(|e| &e == rs)
                            .unwrap_or(false)
                })
        })
        .cloned()
}

fn uints(v: &tensorfs_core::canon::Value) -> tensorfs_core::err::Result<Vec<u64>> {
    tensorfs_core::canon::as_arr("row", "shape", v)?
        .iter()
        .map(|x| tensorfs_core::canon::as_uint("row", "shape", x))
        .collect()
}

/// A part whose GEOMETRY is real and whose bytes are not here. The digest is derived from
/// the key so the document is stable and reproducible; it addresses nothing.
fn placeholder_part(
    key: &str,
    role: &str,
    dtype: Dtype,
    shape: Vec<u64>,
    nbytes: u64,
) -> tensorfs_core::header::Part {
    use tensorfs_core::header::{Body, Part};
    if Part::is_inline(nbytes) {
        return Part {
            dtype,
            shape,
            body: Body::Inline(vec![0u8; nbytes as usize]),
        };
    }
    let grid = limits::GRID_BYTES;
    let n = nbytes.div_ceil(grid);
    let segments = (0..n)
        .map(|i| ObjectRef {
            sha256: tensorfs_core::sha256::hex_digest(
                format!("placeholder:{key}#{role}:{i}").as_bytes(),
            ),
            length: if i + 1 == n { nbytes - i * grid } else { grid },
        })
        .collect();
    Part {
        dtype,
        shape,
        body: Body::Segments(segments),
    }
}

fn emit<D: Plain>(d: &D, flags: &Flags) {
    emit_bytes(&d.canonical_bytes(), flags);
}

fn emit_bytes(bytes: &[u8], flags: &Flags) {
    if let Some(p) = flag(flags, "out") {
        if let Err(e) = fs::write(p, bytes) {
            eprintln!("REFUSED IO_FAILED: write {p}: {e}");
        }
    }
}

/// `check_compatibility` over two documents on disk. Zero tensor bytes are read: the whole
/// decision is a function of a code topology, a header and a closure, which is exactly why
/// it is decidable before a download.
///
/// `--json` prints `{"details":[...],"report":"sha256:...","satisfied":bool}` and nothing
/// else. That is the surface a program should read; the human print below is for people.
pub fn cmd_check(code_p: &Path, header_p: &Path, flags: &Flags) -> ExitCode {
    let code = ok!(CodeTopology::parse(&rd!(code_p)));
    let header = ok!(Header::parse(&rd!(header_p)));
    let mut closure = tensorfs_core::header::Closure::default();
    for p in flag(flags, "closure").unwrap_or("").split(',') {
        if p.is_empty() {
            continue;
        }
        closure.insert(ok!(EncodingSpec::decode_nested(&rd!(Path::new(p)))));
    }
    let pin: Vec<String> = match flag(flags, "pin") {
        Some(p) if !p.is_empty() => p.split(',').map(|s| s.to_string()).collect(),
        // The default pin is the platform namespace plus whatever the closure carries: a
        // local check is not a place to widen border policy.
        _ => {
            let mut v = registry::platform_digests();
            v.extend(closure.specs.iter().map(|(id, _)| id.clone()));
            v
        }
    };
    let lane = ok!(Lane::parse(flag(flags, "lane").unwrap_or("canonical")));
    let t = Instant::now();
    let out = ok!(compat::check_compatibility(
        &code,
        &header,
        &closure,
        &pin,
        lane,
        &markers()
    ));
    let us = t.elapsed().as_secs_f64() * 1e6;
    let report = ok!(CompatibilityReport::of(&code, &header, lane, &out));
    emit(&report, flags);

    // The machine surface. A caller that shells out to this command must not have to
    // recognise a verdict by matching a column-aligned print; tensorhub did exactly that
    // and would have failed closed on any reformat.
    if flag_on(flags, "json") {
        let details = match &out.satisfaction {
            Satisfaction::Satisfied => Vec::new(),
            Satisfaction::Mismatch(d) => vec![tensorfs_core::canon::Value::obj(vec![
                ("code", tensorfs_core::canon::Value::str(d.code.as_str())),
                (
                    "component",
                    tensorfs_core::canon::Value::str(d.component.clone()),
                ),
                ("key", tensorfs_core::canon::Value::str(d.key.clone())),
                (
                    "required",
                    tensorfs_core::canon::Value::str(side(&d.required)),
                ),
                (
                    "present",
                    tensorfs_core::canon::Value::str(side(&d.present)),
                ),
            ])],
        };
        let doc = tensorfs_core::canon::Value::obj(vec![
            (
                "satisfied",
                tensorfs_core::canon::Value::Bool(details.is_empty()),
            ),
            ("details", tensorfs_core::canon::Value::arr(details)),
            (
                "report",
                tensorfs_core::canon::Value::str(report.object_id()),
            ),
        ]);
        println!(
            "{}",
            String::from_utf8_lossy(&tensorfs_core::canon::write(&doc))
        );
        return ExitCode::SUCCESS;
    }

    println!("lane         {}", lane.name());
    println!(
        "topology     required {} / present {} — {}",
        &report.topology_required[7..19],
        &report.topology_present[7..19],
        if report.topology_required == report.topology_present {
            "the fast-path index HITS (an index, never the decision)"
        } else {
            "index miss; the decision is the document comparison below"
        }
    );
    match &out.satisfaction {
        Satisfaction::Satisfied => println!("verdict      Satisfied"),
        Satisfaction::Mismatch(d) => {
            println!("verdict      Mismatch {}", d.code.as_str());
            println!("  first      {}/{}", d.component, d.key);
            println!("  required   {}", side(&d.required));
            println!("  present    {}", side(&d.present));
        }
    }
    for f in &report.facts {
        println!(
            "  facts {:<18} required {:>5} keys {:<28} present {:>5} keys {}",
            f.component,
            f.required_keys,
            dth(&f.required_dtypes),
            f.present_keys,
            dth(&f.present_dtypes)
        );
        for e in &f.present_encodings {
            println!(
                "        {:<12} {:>5} keys  roles {}",
                &e.encoding[7..19],
                e.keys,
                e.roles
                    .iter()
                    .map(|r| format!("{}:{}/rank{}", r.role, r.carrier.name(), r.rank))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    if !out.unconsumed.is_empty() {
        println!(
            "unconsumed   {} key(s) warned-ignored-confessed, first {:?}",
            out.unconsumed.len(),
            out.unconsumed[0]
        );
    }
    println!("report       {}", report.object_id());
    println!(
        "in {us:.0} us, peak RSS {:.1} MiB",
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

fn side(s: &compat::Side) -> String {
    let mut v = Vec::new();
    if let Some(d) = s.dtype {
        v.push(d.name().to_string());
    }
    if let Some(sh) = &s.shape {
        v.push(format!("{sh:?}"));
    }
    if let Some(e) = &s.encoding {
        v.push(e[7..19].to_string());
    }
    if let Some(a) = &s.accepts {
        v.push(format!(
            "accepts {}",
            if a.len() > 20 { &a[..20] } else { a }
        ));
    }
    if v.is_empty() {
        "ABSENT".to_string()
    } else {
        v.join(" ")
    }
}

fn dth(h: &[(Dtype, u64)]) -> String {
    h.iter()
        .map(|(d, n)| format!("{}x{n}", d.name()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A release cut takes only a fully classified candidate.
pub fn cmd_release_cut(report_p: &Path) -> ExitCode {
    let report = ok!(CompatibilityReport::parse(&rd!(report_p)));
    ok!(report.release_cut());
    println!(
        "release cut  OK — {} component(s), zero unclassified extras",
        report.facts.len()
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ registration

/// The REGISTRATION ACT: run A -> B -> A and refuse anything but byte identity. There is no
/// flag that skips it.
pub fn cmd_register(name: &str, kind: &str, spec: &str, flags: &Flags) -> ExitCode {
    let mapping = match kind {
        "widen" => {
            let (a, b) = match spec.split_once(':') {
                Some(x) => x,
                None => {
                    eprintln!("REFUSED MISSING_FIELD: widen takes <from>:<to>");
                    return ExitCode::FAILURE;
                }
            };
            Mapping::Widen {
                from: ok!(Dtype::parse(a)),
                to: ok!(Dtype::parse(b)),
            }
        }
        "rename" => Mapping::Rename {
            pairs: spec
                .split(',')
                .filter_map(|p| p.split_once(':'))
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        },
        "regroup" => Mapping::Regroup {
            parts: spec.split(',').filter_map(|p| p.parse().ok()).collect(),
        },
        o => {
            eprintln!("REFUSED UNKNOWN_FIELD: mapping kind {o:?}");
            return ExitCode::FAILURE;
        }
    };
    let corpus: Vec<Vec<u8>> = match flag(flags, "corpus") {
        Some(p) => vec![rd!(Path::new(p))],
        // The default for a 16-bit source is the COMPLETE domain: every representable
        // input, so the registration is a proof rather than a sample.
        None => match &mapping {
            Mapping::Widen { from, .. } if from.size() == 2 => grade::complete_domain_16(),
            Mapping::Rename { pairs } => vec![pairs
                .iter()
                .map(|(a, _)| format!("{a}suffix.weight"))
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
                .into_iter()
                .chain([b'\n'])
                .collect()],
            Mapping::Regroup { parts } => {
                let n: u64 = parts.iter().sum();
                vec![(0..n).map(|i| (i % 251) as u8).collect()]
            }
            _ => {
                eprintln!("REFUSED MISSING_FIELD: this mapping needs an explicit --corpus");
                return ExitCode::FAILURE;
            }
        },
    };
    let t = Instant::now();
    let reg = ok!(ConversionRegistration::register(name, mapping, &corpus));
    emit_bytes(&reg.canonical_bytes(), flags);
    println!("registered   {} — {}", reg.name, reg.mapping.describe());
    println!(
        "round trip   {} case(s), {} B, coverage {} in {:.1} ms",
        reg.round_trip.cases,
        reg.round_trip.bytes,
        reg.round_trip.coverage,
        t.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "registration {}",
        tensorfs_core::ids::object_id(&reg.canonical_bytes())
    );
    ExitCode::SUCCESS
}

fn load_registry(flags: &Flags) -> Result<Registry, ExitCode> {
    let mut r = Registry::default();
    for p in flag(flags, "registry").unwrap_or("").split(',') {
        if p.is_empty() {
            continue;
        }
        let b = read(Path::new(p))?;
        match ConversionRegistration::parse(&b) {
            Ok(x) => r.registrations.push(x),
            Err(e) => {
                eprintln!("REFUSED {e}");
                return Err(ExitCode::FAILURE);
            }
        }
    }
    Ok(r)
}

/// The hub's side of a decision document: parse it, or refuse. th-002 reads decisions and
/// must never build a second reader for them.
pub fn cmd_decision(p: &Path) -> ExitCode {
    let d = ok!(GradeDecision::parse(&rd!(p)));
    println!(
        "decision     {}",
        tensorfs_core::ids::object_id(&d.canonical_bytes())
    );
    println!("report       {}", d.report);
    println!("grade        {}", d.grade.name());
    ExitCode::SUCCESS
}

/// The machine surface of a grade. TOTAL on purpose: every key is present on every grade,
/// empty where that grade does not carry it. A consumer that fails closed must be able to
/// decode one shape and read `grade`, not discover which keys this variant happened to
/// emit — and tensorhub decodes with unknown fields refused, so an absent key and a new key
/// are equally fatal there.
///
/// `remedy_*` is the whole point of the four levels: PRODUCIBLE names the encoding/dtype
/// the CONTRACT requires, so "we could run this if you convert it to X" is distinguishable
/// from "no".
fn grade_json(decision: &GradeDecision) -> Vec<u8> {
    let (registration, mapping, target, source, reason) = match &decision.grade {
        Grade::Compatible => ("", "", "", "", ""),
        Grade::Convertible {
            registration,
            mapping,
        } => (registration.as_str(), mapping.as_str(), "", "", ""),
        Grade::Producible { target, source } => ("", "", target.as_str(), source.as_str(), ""),
        Grade::Incompatible { reason } => ("", "", "", "", reason.as_str()),
    };
    let v = tensorfs_core::canon::Value::obj(vec![
        (
            "decision",
            tensorfs_core::canon::Value::str(tensorfs_core::ids::object_id(
                &decision.canonical_bytes(),
            )),
        ),
        (
            "grade",
            tensorfs_core::canon::Value::str(decision.grade.name()),
        ),
        (
            "report",
            tensorfs_core::canon::Value::str(decision.report.clone()),
        ),
        (
            "remedy_registration",
            tensorfs_core::canon::Value::str(registration),
        ),
        ("remedy_mapping", tensorfs_core::canon::Value::str(mapping)),
        ("remedy_target", tensorfs_core::canon::Value::str(target)),
        ("remedy_source", tensorfs_core::canon::Value::str(source)),
        ("reason", tensorfs_core::canon::Value::str(reason)),
    ]);
    tensorfs_core::canon::write(&v)
}

/// The hub's grading, over a report. Both sides are already named in the report, so this
/// reads no header, no store and no network.
///
/// `--json` prints one `grade_json` object and nothing else — the surface a program reads.
/// The column-aligned print below is for people, and no caller may match on it.
pub fn cmd_grade(report_p: &Path, flags: &Flags) -> ExitCode {
    let report = ok!(CompatibilityReport::parse(&rd!(report_p)));
    let reg = match load_registry(flags) {
        Ok(r) => r,
        Err(c) => return c,
    };
    // Grading over a report ALONE cannot prove a registration closes the whole gap — only
    // `verify-compat`, which holds both documents, can. `--prove-with` supplies them here.
    let t = Instant::now();
    let g = match (flag(flags, "code"), flag(flags, "header")) {
        (Some(c), Some(h)) => {
            let code = ok!(CodeTopology::parse(&rd!(Path::new(c))));
            let header = ok!(Header::parse(&rd!(Path::new(h))));
            let mut closure = tensorfs_core::header::Closure::default();
            for p in flag(flags, "closure").unwrap_or("").split(',') {
                if !p.is_empty() {
                    closure.insert(ok!(EncodingSpec::decode_nested(&rd!(Path::new(p)))));
                }
            }
            let pin: Vec<String> = match flag(flags, "pin") {
                Some(p) if !p.is_empty() => p.split(',').map(|s| s.to_string()).collect(),
                _ => header
                    .encodings
                    .iter()
                    .map(EncodingSpec::object_id)
                    .collect(),
            };
            ok!(grade::resolve(
                &code,
                &header,
                &closure,
                &pin,
                report.lane,
                &markers(),
                &reg
            ))
            .1
        }
        _ => {
            let g = grade::grade(&report, &reg);
            ok!(grade::guard_over_constraint(&report, &reg, &g));
            g
        }
    };
    let us = t.elapsed().as_secs_f64() * 1e6;
    let decision = GradeDecision {
        report: report.object_id(),
        grade: g.clone(),
    };
    emit_bytes(&decision.canonical_bytes(), flags);
    if flag_on(flags, "json") {
        println!("{}", String::from_utf8_lossy(&grade_json(&decision)));
        return ExitCode::SUCCESS;
    }
    println!("grade        {}", g.name());
    match &g {
        Grade::Compatible => {}
        Grade::Convertible {
            registration,
            mapping,
        } => println!("  registered {registration} — {mapping}"),
        Grade::Producible { target, source } => {
            println!("  target     {target}   (from the CONTRACT, never the observation)");
            println!("  source     {source}");
        }
        Grade::Incompatible { reason } => println!("  reason     {reason}"),
    }
    println!(
        "decision     {}",
        tensorfs_core::ids::object_id(&decision.canonical_bytes())
    );
    println!(
        "in {us:.0} us over {} registration(s)",
        reg.registrations.len()
    );
    ExitCode::SUCCESS
}

// ------------------------------------------------------------------ the hermetic verifier

/// Build a request out of loose documents, so the seam can be driven by hand.
pub fn cmd_request(code_p: &Path, header_p: &Path, flags: &Flags) -> ExitCode {
    let code_topology = rd!(code_p);
    let header_bytes = rd!(header_p);
    let header = ok!(Header::parse(&header_bytes));
    let mut closure = Vec::new();
    for p in flag(flags, "closure").unwrap_or("").split(',') {
        if !p.is_empty() {
            closure.push(rd!(Path::new(p)));
        }
    }
    let mut registrations = Vec::new();
    for p in flag(flags, "registry").unwrap_or("").split(',') {
        if !p.is_empty() {
            registrations.push(rd!(Path::new(p)));
        }
    }
    let pin: Vec<String> = match flag(flags, "pin") {
        Some(p) if !p.is_empty() => p.split(',').map(|s| s.to_string()).collect(),
        _ => header
            .encodings
            .iter()
            .map(EncodingSpec::object_id)
            .collect(),
    };
    let req = VerifyRequest {
        code_topology,
        header: header_bytes,
        closure,
        registrations,
        lane: ok!(Lane::parse(flag(flags, "lane").unwrap_or("canonical"))),
        pin,
        known_markers: markers(),
    };
    emit(&req, flags);
    println!("request      {}", req.object_id());
    println!("bytes        {}", req.canonical_bytes().len());
    ExitCode::SUCCESS
}

/// The verifier process. One file in, one file out, written whole or not at all.
pub fn cmd_verify(req_p: &Path, out_p: &Path, flags: &Flags) -> ExitCode {
    let bytes = rd!(req_p);
    if bytes.len() > limits::DOC_MAX_BYTES {
        eprintln!("REFUSED SIZE_CAP: request over the document cap");
        return ExitCode::FAILURE;
    }
    let t = Instant::now();
    let response = ok!(verifier::run(&bytes));
    // A dev-only truncation point, so the CONSUMER's partial-output refusal can be armed
    // against a real short file rather than described.
    let response = match flag(flags, "truncate-at") {
        Some(n) => response[..n.parse::<usize>().unwrap_or(0).min(response.len())].to_vec(),
        None => response,
    };
    let tmp = out_p.with_extension("partial");
    if let Err(e) = fs::write(&tmp, &response).and_then(|_| fs::rename(&tmp, out_p)) {
        eprintln!("REFUSED IO_FAILED: write response: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "verified     {} B response in {:.2} ms, peak RSS {:.1} MiB",
        response.len(),
        t.elapsed().as_secs_f64() * 1000.0,
        rss_kb("VmHWM") as f64 / 1024.0
    );
    ExitCode::SUCCESS
}

/// The hub's side: consume a response, or refuse. Nothing between those two.
pub fn cmd_consume(req_p: &Path, resp_p: &Path) -> ExitCode {
    let req = rd!(req_p);
    let resp = fs::read(resp_p).unwrap_or_default();
    let (report, decision) = ok!(verifier::consume(&req, &resp));
    println!("report       {}", report.object_id());
    println!("lane         {}", report.lane.name());
    match &report.satisfaction {
        Satisfaction::Satisfied => println!("satisfaction Satisfied"),
        Satisfaction::Mismatch(d) => println!(
            "satisfaction Mismatch {} at {}/{}",
            d.code.as_str(),
            d.component,
            d.key
        ),
    }
    println!("grade        {}", decision.grade.name());
    ExitCode::SUCCESS
}
