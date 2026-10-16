//! PRODUCER-MINED vector rows (scripts/mine-vectors.py) into the canonical vector document
//! `vectors/mined/<name>.json`:
//!
//!     cargo run -p tensorfs-core --example mine-vectors -- <rows> <name> [<match>]

use std::process::ExitCode;

use tensorfs_core::dtype::Dtype;
use tensorfs_core::vectors::{Case, EncodingVectors};

macro_rules! ok {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                eprintln!("REFUSED {e}");
                return ExitCode::FAILURE;
            }
        }
    };
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [rows, name] => mine(std::path::Path::new(rows), name, ""),
        [rows, name, want] => mine(std::path::Path::new(rows), name, want),
        _ => {
            eprintln!("usage: mine-vectors <rows> <name> [<match>]");
            ExitCode::from(2)
        }
    }
}

/// Turn PRODUCER-MINED rows into the canonical vector document. The rows carry raw bytes
/// and nothing else; canonicalisation happens here, once, in the one writer.
fn mine(rows: &std::path::Path, name: &str, want: &str) -> ExitCode {
    let text = match std::fs::read_to_string(rows) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", rows.display());
            return ExitCode::FAILURE;
        }
    };
    let shape = |s: &str| -> Vec<u64> {
        if s.is_empty() {
            vec![]
        } else {
            s.split(',').filter_map(|d| d.parse().ok()).collect()
        }
    };

    let mut cases: Vec<Case> = Vec::new();
    let mut cur: Option<Case> = None;
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first().copied() {
            Some("case") => {
                if let Some(c) = cur.take() {
                    cases.push(c);
                }
                cur = Some(Case {
                    name: t[1].to_string(),
                    logical_dtype: Dtype::F32,
                    logical_shape: vec![],
                    roles: vec![],
                    expect_logical_bits: vec![],
                });
            }
            Some("logical") => {
                if let Some(c) = cur.as_mut() {
                    c.logical_dtype = ok!(Dtype::parse(t[1]));
                    c.logical_shape = shape(t.get(2).copied().unwrap_or(""));
                }
            }
            Some("expect") => {
                if let Some(c) = cur.as_mut() {
                    c.expect_logical_bits = ok!(tensorfs_core::b64::decode(t[1]));
                }
            }
            Some("role") => {
                if let Some(c) = cur.as_mut() {
                    let (rname, dt) = (t[1].to_string(), ok!(Dtype::parse(t[2])));
                    // a rank-0 role prints an empty shape field, so the payload shifts left
                    let (sh, b64) = if t.len() >= 5 {
                        (shape(t[3]), t[4])
                    } else {
                        (vec![], t[3])
                    };
                    c.roles.push((
                        rname,
                        tensorfs_core::vectors::RoleBits {
                            dtype: dt,
                            shape: sh,
                            bits: ok!(tensorfs_core::b64::decode(b64)),
                        },
                    ));
                }
            }
            _ => {}
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }
    cases.retain(|c| c.name.contains(want));
    for c in cases.iter_mut() {
        c.roles.sort_by(|a, b| a.0.cmp(&b.0));
    }
    if cases.is_empty() {
        eprintln!("no mined case matches {want:?}");
        return ExitCode::from(2);
    }
    let v = EncodingVectors { cases };
    let bytes = v.fixture_bytes();
    if let Err(e) = EncodingVectors::parse_fixture(&bytes) {
        {
        eprintln!("REFUSED {e}");
        return ExitCode::FAILURE;
    }
    }
    let d: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vectors/mined")
        .components()
        .collect();
    if let Err(e) = std::fs::create_dir_all(&d) {
        eprintln!("{}: {e}", d.display());
        return ExitCode::FAILURE;
    }
    let path = d.join(format!("{name}.json"));
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("{}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    println!("mined {} ({} B)", path.display(), bytes.len());
    println!("  cases     {}", v.cases.len());
    println!(
        "  fixture_id {}",
        tensorfs_core::ids::object_id(&v.fixture_bytes())
    );
    for c in &v.cases {
        println!(
            "    {:<34} logical {} {:?}, roles {:?}",
            c.name,
            c.logical_dtype.name(),
            c.logical_shape,
            c.roles.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
        );
    }
    ExitCode::SUCCESS
}

