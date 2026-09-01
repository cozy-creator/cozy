//! The layout plane's CLI (tfs-009): `check_compatibility` over two documents on disk.
//!
//! This file is a CONSUMER of `compat`. It prints a verdict; it never decides one.

use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use tensorfs_core::checkpoint as ck;
use tensorfs_core::compat::{
    self, Accepts, CodeTopology, CompatibilityReport, Lane, Requirement, Satisfaction,
};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::Header;
use tensorfs_core::ids::{ObjectRef, Plain};
use tensorfs_core::ingest::fingerprint::MARKER_SUFFIXES;
use tensorfs_core::limits;
use tensorfs_core::registry;
use tensorfs_core::spec::EncodingSpec;
use tensorfs_core::store::Store;

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
/// transcriber that reads a pinned foreign header (`scripts/h3-pairs.py`) must never become
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
