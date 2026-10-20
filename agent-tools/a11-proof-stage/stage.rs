//! One-off metadata client. Built against the approved A11 tree, never copied core code.
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tensorfs::{
    failure::{Failure, Kind},
    format::{Header, ObjectRef, Tree},
    import::remote,
    migrate::{
        d10::{Migrator, UPSTREAMS},
        v1,
    },
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const APPROVED: &str = "e2c8ded23669b7b18185808c1c1c0934693def3b";
const CAP: u64 = 32 << 20;
#[derive(Deserialize)]
struct Bundle {
    repos: Vec<Value>,
    objects: BTreeMap<String, ObjectRef>,
}
fn refuse(detail: impl Into<String>) -> Failure {
    Failure {
        kind: Kind::Invalid,
        detail: detail.into(),
    }
}
fn read(root: &Path, bundle: &Bundle, at: &ObjectRef) -> tensorfs::failure::Result<Vec<u8>> {
    if at.length > CAP || bundle.objects.get(&at.sha256) != Some(at) {
        return Err(refuse(format!("not allowlisted metadata: {}", at.sha256)));
    }
    let bytes = fs::read(root.join("objects").join(&at.sha256))
        .map_err(|_| refuse(format!("missing metadata: {}", at.sha256)))?;
    if ObjectRef::of(&bytes) != *at {
        return Err(refuse("metadata identity mismatch"));
    }
    Ok(bytes)
}
fn segment(s: &str) -> Result<&str> {
    if s.is_empty()
        || s == "."
        || s == ".."
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b))
    {
        return Err("unsafe output segment".into());
    }
    Ok(s)
}
fn run(command: &str, path: &Path, output: Option<&Path>) -> Result<Value> {
    if option_env!("A11_NATIVE_SOURCE") != Some(APPROVED) {
        return Err("client was not built against approved A11".into());
    }
    let bundle: Bundle = serde_json::from_slice(&fs::read(path)?)?;
    let root = path.parent().ok_or("bundle has no parent")?;
    if command == "inventory" {
        let mut assets = BTreeMap::new();
        for repo in &bundle.repos {
            for release in repo["releases"].as_array().ok_or("missing releases")? {
                for value in release["lanes"]
                    .as_object()
                    .ok_or("missing lanes")?
                    .values()
                {
                    let at: ObjectRef = serde_json::from_value(value.clone())?;
                    let raw = read(root, &bundle, &at)?;
                    if let Some(manifest) = v1::manifest(&raw)? {
                        for asset in v1::assets(&read(root, &bundle, &manifest.header)?)? {
                            assets.insert(
                                asset.sha256.clone(),
                                json!({"object": asset, "manifest": at.sha256}),
                            );
                        }
                    }
                }
            }
        }
        return Ok(json!({"assets": assets, "tensor_chunk_reads": 0, "a11_source": APPROVED}));
    }
    if command != "stage-only" {
        return Err("command must be inventory or stage-only".into());
    }
    let output = output.ok_or("stage-only requires a fresh output directory")?;
    fs::create_dir(output)?;
    fs::create_dir(output.join("objects"))?;
    let token = std::env::var("A11_HF_TOKEN").unwrap_or_default();
    let access = remote::Access {
        huggingface_token: if token.is_empty() {
            token
        } else {
            format!("bearer {token}")
        },
        ..Default::default()
    };
    let mut migrator = Migrator::new(access, UPSTREAMS);
    let reads = RefCell::new(BTreeSet::new());
    let source = |at: &ObjectRef| {
        let bytes = read(root, &bundle, at)?;
        reads.borrow_mut().insert(at.sha256.clone());
        Ok(bytes)
    };
    let mut staged = Vec::new();
    for doc in &bundle.repos {
        let migrated = migrator.repo(&serde_json::to_vec(doc)?, &source)?;
        let mut objects: BTreeMap<_, _> = migrated
            .objects
            .iter()
            .map(|(at, bytes)| (at.sha256.clone(), bytes.clone()))
            .collect();
        for (at, bytes) in &migrated.objects {
            if ObjectRef::of(bytes) != *at {
                return Err("A11 output identity mismatch".into());
            }
            fs::write(output.join("objects").join(&at.sha256), bytes)?;
        }
        let repo_name = migrated.repo.name.clone();
        let repo_path = repo_name
            .split('/')
            .try_fold(output.join("repos"), |path, name| -> Result<PathBuf> {
                Ok(path.join(segment(name)?))
            })?;
        fs::create_dir_all(&repo_path)?;
        fs::write(repo_path.join("repo.json"), migrated.repo.to_json()?)?;
        for release in &migrated.repo.releases {
            if !objects.contains_key(&release.tree.sha256) {
                objects.insert(release.tree.sha256.clone(), source(&release.tree)?);
            }
            let tree = Tree::from_json(
                objects
                    .get(&release.tree.sha256)
                    .ok_or("missing staged tree")?,
            )?;
            for (variant, at) in tree.variants() {
                if !objects.contains_key(&at.sha256) {
                    objects.insert(at.sha256.clone(), source(at)?);
                }
                let raw = objects.get(&at.sha256).ok_or("missing staged header")?;
                let header = Header::from_cbor(raw)?;
                let folder = repo_path.join(segment(&release.version)?).join(segment(
                    if variant.is_empty() {
                        "default"
                    } else {
                        variant
                    },
                )?);
                fs::create_dir_all(folder.join("chunks"))?;
                fs::write(folder.join("model.cozytensors"), raw)?;
                let mut metadata_objects = BTreeMap::new();
                for (_, _, asset) in header.assets() {
                    metadata_objects.insert(asset.sha256.clone(), asset.clone());
                }
                for (_, asset) in tree.repo_files() {
                    metadata_objects.insert(asset.sha256.clone(), asset.clone());
                }
                // A11 may repack tiny parts already inline in the original header.
                // Original tensor chunks are never read; only newly generated packs exist here.
                let generated_packs: Vec<_> = header
                    .chunks
                    .iter()
                    .filter(|chunk| objects.contains_key(&chunk.sha256))
                    .cloned()
                    .collect();
                for chunk in &generated_packs {
                    metadata_objects.insert(chunk.sha256.clone(), chunk.clone());
                }
                for asset in metadata_objects.values() {
                    let bytes = match objects.get(&asset.sha256) {
                        Some(bytes) => bytes.clone(),
                        None => source(asset)?,
                    };
                    if ObjectRef::of(&bytes) != *asset {
                        return Err("asset identity mismatch".into());
                    }
                    fs::write(folder.join("chunks").join(&asset.sha256), bytes)?;
                }
                for (name, asset) in tree.repo_files() {
                    let target = name
                        .split('/')
                        .try_fold(folder.clone(), |path, name| -> Result<PathBuf> {
                            Ok(path.join(segment(name)?))
                        })?;
                    fs::create_dir_all(target.parent().ok_or("asset has no parent")?)?;
                    let bytes = fs::read(folder.join("chunks").join(&asset.sha256))?;
                    fs::write(target, bytes)?;
                }
                staged.push(json!({"repo": repo_name, "release": release.version, "variant": variant, "header": at,
                    "tree": release.tree, "folder": folder, "tensors": header.tensors().count(), "chunks": header.chunks,
                    "source": header.metadata.get("source"), "generated_inline_packs": generated_packs}));
            }
        }
        fs::write(
            repo_path.join("a11-report.json"),
            serde_json::to_vec_pretty(&migrated.report)?,
        )?;
    }
    let receipt = json!({"a11_source": APPROVED, "source_metadata_reads": reads.into_inner(), "staged": staged,
        "tensor_chunk_downloads": 0, "hub_mutations": 0});
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(receipt)
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [command, bundle] if command == "inventory" => run(command, Path::new(bundle), None),
        [command, bundle, output] if command == "stage-only" => {
            run(command, Path::new(bundle), Some(Path::new(output)))
        }
        _ => Err("usage: a11_proof_stage inventory BUNDLE | stage-only BUNDLE NEW_OUTPUT".into()),
    };
    match result {
        Ok(receipt) => println!("{receipt}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
