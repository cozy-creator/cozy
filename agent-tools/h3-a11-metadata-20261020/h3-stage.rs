//! One-off A11 client. Compile as a temporary tensorfs example, never as a public API.
//! Input is an audited metadata-only bundle; HF requests go only to its loopback replay.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};
use tensorfs::failure::{Failure, Kind, Result};
use tensorfs::format::{Header, ObjectRef, Tree};
use tensorfs::import::remote;
use tensorfs::migrate::d10::{Migrator, UPSTREAMS};
use tensorfs::store2::Store;

const BASE: &str = "25a1d2f6bfbf07e84f06a110b30083b9fb5a45f2";
const BUNDLE_SHA256: &str = "891aeb30592b74beedba96583b7bd130b6730b07fdba8977a23f4d0248811dff";
const CAP: u64 = 32 << 20;

#[derive(Deserialize)]
struct Bundle {
    format: String,
    source_base: String,
    objects: BTreeMap<String, ObjectRef>,
    repos: Vec<Value>,
}

fn bad(kind: Kind, detail: impl Into<String>) -> Failure {
    Failure {
        kind,
        detail: detail.into(),
    }
}

fn io(error: std::io::Error) -> Failure {
    bad(Kind::Io, error.to_string())
}

fn run(bundle_path: &Path, destination: &Path, upstream: &str) -> Result<Value> {
    // Only the Python controller's ephemeral loopback replay is accepted, never live HF.
    let port = upstream
        .strip_prefix("http://127.0.0.1:")
        .ok_or_else(|| bad(Kind::Invalid, "upstream must be the loopback bundle replay"))?;
    if port.parse::<u16>().ok().filter(|port| *port > 0).is_none() {
        return Err(bad(Kind::Invalid, "invalid replay port"));
    }
    if fs::symlink_metadata(bundle_path).map_err(io)?.len() > 1 << 20 {
        return Err(bad(Kind::Invalid, "bundle document exceeds 1 MiB"));
    }
    let bytes = fs::read(bundle_path).map_err(io)?;
    if ObjectRef::of(&bytes).sha256 != BUNDLE_SHA256 {
        return Err(bad(Kind::Integrity, "input is not the exact collected H3 metadata bundle"));
    }
    let bundle: Bundle =
        serde_json::from_slice(&bytes).map_err(|e| bad(Kind::Invalid, e.to_string()))?;
    if bundle.format != "cozy.a11.metadata-bundle/1"
        || bundle.source_base != BASE
        || bundle.repos.len() != 1
    {
        return Err(bad(
            Kind::Invalid,
            "unexpected bundle format, source base or repo count",
        ));
    }
    let directory = bundle_path
        .parent()
        .ok_or_else(|| bad(Kind::Invalid, "bundle has no parent"))?
        .join("objects");
    let reads = RefCell::new(BTreeSet::new());
    let read = |object: &ObjectRef| -> Result<Vec<u8>> {
        if bundle.objects.get(&object.sha256) != Some(object) || object.length > CAP {
            return Err(bad(
                Kind::Denied,
                "migrator requested an object outside the metadata allowlist",
            ));
        }
        let path = directory.join(&object.sha256);
        let stat = fs::symlink_metadata(&path).map_err(io)?;
        if !stat.is_file() || stat.len() != object.length {
            return Err(bad(
                Kind::Integrity,
                "source metadata file type or length changed",
            ));
        }
        let bytes = fs::read(path).map_err(io)?;
        if ObjectRef::of(&bytes) != *object {
            return Err(bad(Kind::Integrity, "source metadata digest changed"));
        }
        reads.borrow_mut().insert(object.sha256.clone());
        Ok(bytes)
    };
    // A fresh destination is mandatory. No existing store or source ref can be updated.
    fs::create_dir(destination).map_err(io)?;
    let store = Store::open(destination).map_err(Failure::from)?;
    let access = remote::Access {
        huggingface: upstream.into(),
        allow_local: true,
        ..remote::Access::default()
    };
    let mut migrator = Migrator::new(access, UPSTREAMS);
    let mut reports = Vec::new();
    let mut objects = BTreeMap::new();
    let mut retained = BTreeMap::new();
    for repo in &bundle.repos {
        let bytes = serde_json::to_vec(repo).map_err(|e| bad(Kind::Invalid, e.to_string()))?;
        let migrated = migrator.repo(&bytes, &read)?;
        for (at, bytes) in &migrated.objects {
            if ObjectRef::of(bytes) != *at {
                return Err(bad(
                    Kind::Integrity,
                    "migrator produced the wrong object digest",
                ));
            }
            let written = store
                .put(bytes.as_slice(), Some(at))
                .map_err(Failure::from)?;
            objects.insert(written.sha256.clone(), written);
        }
        for release in &migrated.repo.releases {
            let tree = Tree::from_json(&store.read(&release.tree).map_err(Failure::from)?)?;
            for (variant, at) in tree.variants() {
                let header = Header::from_cbor(&store.read(at).map_err(Failure::from)?)?;
                if variant != "int8" || header.chunks.len() != 1676
                    || header.components().iter().map(|c| c.tensors.len()).sum::<usize>() != 3858
                {
                    return Err(bad(Kind::Integrity, "unexpected selected H3 header geometry"));
                }
                retained.insert(
                    format!("{}@{}:{variant}", migrated.repo.name, release.version),
                    header.chunks,
                );
            }
        }
        store
            .update_ref(&migrated.repo.name, |_| Some(migrated.repo.clone()))
            .map_err(Failure::from)?;
        reports.push(migrated.report);
    }
    let chunks: BTreeSet<_> = retained
        .values()
        .flat_map(|chunks| chunks.iter().cloned())
        .collect();
    for chunk in &chunks {
        if store.path(chunk).map_err(Failure::from)?.exists() {
            return Err(bad(
                Kind::Integrity,
                "a retained tensor chunk was materialized in the metadata-only destination",
            ));
        }
    }
    Ok(json!({
        "format": "cozy.a11.metadata-stage-receipt/1", "source_base": BASE,
        "native_version": env!("CARGO_PKG_VERSION"), "native_source": option_env!("A11_NATIVE_SOURCE"), "selection": "one exact H3 int8 lane; no live update",
        "source_metadata_reads": reads.into_inner(), "staged_objects": objects,
        "retained_chunks": retained, "retained_chunk_count": chunks.len(),
        "chunk_bytes_downloaded": 0, "source_mutations": 0, "reports": reports,
        "destination": destination
    }))
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        eprintln!("usage: h3_metadata_stage BUNDLE_JSON FRESH_DESTINATION http://127.0.0.1:PORT");
        std::process::exit(2);
    }
    match run(&args[0], &args[1], &args[2].to_string_lossy()) {
        Ok(receipt) => println!("{receipt}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
