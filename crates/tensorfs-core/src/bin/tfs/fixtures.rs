//! `tfs structure` — bank and match structure fixtures (tfs-008).
//!
//! Banking has exactly two sources and no third: TRANSCRIBED rows from real evidence, or a
//! live header this repo actually wrote. Both use the same topology projection as a header,
//! so a fixture's `topology_digest` is directly comparable with a checkpoint's.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tensorfs_core::checkpoint as ck;
use tensorfs_core::dtype::Dtype;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::structure::{Logical, Provenance, StructureFixture, REAL, SYNTHETIC};

use crate::{bail, flag, Flags};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return bail(e),
        }
    };
}

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vectors/structure")
        .components()
        .collect()
}

fn provenance(f: &Flags) -> Result<Provenance, ExitCode> {
    let kind = flag(f, "provenance").unwrap_or(SYNTHETIC).to_string();
    let source = flag(f, "source").unwrap_or("").to_string();
    let note = flag(f, "note").unwrap_or("").to_string();
    let source_sha256 = flag(f, "source-sha256").map(|s| s.to_string());
    if kind != REAL && kind != SYNTHETIC {
        eprintln!("--provenance is {REAL} or {SYNTHETIC}");
        return Err(ExitCode::from(2));
    }
    Ok(Provenance {
        kind,
        source,
        source_sha256,
        note,
    })
}

fn write_fixture(fx: &StructureFixture) -> ExitCode {
    let d = dir();
    if let Err(e) = std::fs::create_dir_all(&d) {
        eprintln!("{}: {e}", d.display());
        return ExitCode::FAILURE;
    }
    let bytes = fx.to_bytes();
    // Round-trip before writing: malformed reviewed data must never reach the bank.
    if let Err(e) = StructureFixture::parse(&bytes) {
        return bail(e);
    }
    let path = d.join(format!("{}-{}.json", fx.task_family, fx.structural_variant));
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("{}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    println!("banked {}", path.display());
    println!("  stamp            {}", fx.stamp());
    println!("  topology_digest  {}", fx.topology_digest());
    println!(
        "  {} components, {} logical keys, {} B",
        fx.components.len(),
        fx.count(),
        bytes.len()
    );
    println!(
        "  provenance       {} — {} {}",
        fx.provenance.kind,
        fx.provenance.source,
        fx.provenance
            .source_sha256
            .as_deref()
            .map(|h| format!("sha256:{h}"))
            .unwrap_or_default()
    );
    ExitCode::SUCCESS
}

/// Bank from TRANSCRIBED rows: `component<TAB>key<TAB>dtype<TAB>d,d,d`.
pub fn cmd_bank_rows(rows: &Path, family: &str, variant: &str, f: &Flags) -> ExitCode {
    let text = match std::fs::read_to_string(rows) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", rows.display());
            return ExitCode::FAILURE;
        }
    };
    let mut components: Vec<(String, Vec<(String, Logical)>)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 4 {
            eprintln!(
                "row {}: expected 4 tab-separated fields, got {}",
                n + 1,
                c.len()
            );
            return ExitCode::FAILURE;
        }
        let dtype = ok!(Dtype::parse(c[2]));
        let shape: Vec<u64> = if c[3].is_empty() {
            vec![]
        } else {
            match c[3].split(',').map(|d| d.parse()).collect() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("row {}: bad shape {:?}", n + 1, c[3]);
                    return ExitCode::FAILURE;
                }
            }
        };
        let leaf = (c[1].to_string(), Logical { dtype, shape });
        match components.iter_mut().find(|(name, _)| name == c[0]) {
            Some((_, ks)) => ks.push(leaf),
            None => components.push((c[0].to_string(), vec![leaf])),
        }
    }
    let prov = match provenance(f) {
        Ok(p) => p,
        Err(c) => return c,
    };
    write_fixture(&StructureFixture {
        task_family: family.to_string(),
        structural_variant: variant.to_string(),
        provenance: prov,
        components,
    })
}

/// Bank from a live header this repo wrote — the projection comes from the real document.
pub fn cmd_bank_header(root: &Path, hex: &str, family: &str, variant: &str, f: &Flags) -> ExitCode {
    let store = ok!(Store::open(root));
    let len = match std::fs::metadata(store.object_path(hex)) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("REFUSED OBJECT_ABSENT: {hex}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let href = ObjectRef {
        sha256: hex.to_string(),
        length: len,
    };
    let header = ok!(ck::load_header(&store, &href));
    let mut components: Vec<(String, Vec<(String, Logical)>)> = Vec::new();
    for (c, k, t) in header.tensors() {
        let leaf = (
            k.clone(),
            Logical {
                dtype: t.dtype,
                shape: t.shape.clone(),
            },
        );
        match components.iter_mut().find(|(n, _)| n == c) {
            Some((_, ks)) => ks.push(leaf),
            None => components.push((c.clone(), vec![leaf])),
        }
    }
    let mut prov = match provenance(f) {
        Ok(p) => p,
        Err(c) => return c,
    };
    if prov.source.is_empty() {
        prov.source = format!("live header sha256:{hex}");
    }
    if prov.kind == REAL && prov.source_sha256.is_none() {
        prov.source_sha256 = Some(hex.to_string());
    }
    let fx = StructureFixture {
        task_family: family.to_string(),
        structural_variant: variant.to_string(),
        provenance: prov,
        components,
    };
    // The banked digest must equal the digest of the header it was banked from.
    let want = header.topology_digest();
    if fx.topology_digest() != want {
        eprintln!(
            "PROJECTION DISAGREEMENT: fixture {} != header {want}",
            fx.topology_digest()
        );
        return ExitCode::FAILURE;
    }
    println!("  projection agrees with the source header: {want}");
    write_fixture(&fx)
}

fn banked() -> Vec<(PathBuf, StructureFixture)> {
    let mut out = Vec::new();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir())
        .map(|d| {
            d.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    for p in paths {
        if let Ok(b) = std::fs::read(&p) {
            match StructureFixture::parse(&b) {
                Ok(fx) => out.push((p, fx)),
                Err(e) => eprintln!("REFUSED {}: {e}", p.display()),
            }
        }
    }
    out
}

pub fn cmd_list() -> ExitCode {
    let all = banked();
    println!(
        "{} banked structure fixtures in {}",
        all.len(),
        dir().display()
    );
    for (_, fx) in &all {
        println!(
            "  {:<26} {}  {} keys  [{}] {}",
            fx.stamp(),
            fx.topology_digest(),
            fx.count(),
            fx.provenance.kind,
            fx.provenance.source
        );
    }
    if all.is_empty() {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Does a real checkpoint's structure match a banked fixture? The comparison is one digest
/// equality, because both sides come from the same projection function.
pub fn cmd_match(root: &Path, hex: &str) -> ExitCode {
    let store = ok!(Store::open(root));
    let len = match std::fs::metadata(store.object_path(hex)) {
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
            length: len
        }
    ));
    let got = header.topology_digest();
    println!("checkpoint topology_digest {got}");
    let all = banked();
    for (_, fx) in &all {
        if fx.topology_digest() == got {
            println!(
                "  MATCH {} ({} keys, provenance {})",
                fx.stamp(),
                fx.count(),
                fx.provenance.kind
            );
            return ExitCode::SUCCESS;
        }
    }
    println!(
        "  NO MATCH among {} banked fixtures — an unknown structure STORES (identity derives) \
         and refuses serving with an upgrade remedy",
        all.len()
    );
    for (_, fx) in &all {
        println!("    {:<26} {}", fx.stamp(), fx.topology_digest());
    }
    ExitCode::SUCCESS
}
