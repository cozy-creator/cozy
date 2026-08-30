use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::canon::Value;
use tensorfs_core::catalog::{Catalog, WriterGuard};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::header::{Body, Header};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{Entry, Manifest};
use tensorfs_core::repository::{self, Mutation, Repository};
use tensorfs_core::storage::{self, Census, InventoryEntry};
use tensorfs_core::store::{Fault, Store};

use crate::{bail, flag, Flags};

pub fn cmd_manifest_build(entries_path: &Path, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <manifest.json> is required");
        return ExitCode::from(2);
    };
    let input_length = match fs::metadata(entries_path) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            return bail(Refusal {
                code: Code::IO_FAILED,
                detail: format!("stat {}: {error}", entries_path.display()),
            })
        }
    };
    if input_length > Manifest::MAX_BYTES as u64 {
        return bail(Refusal {
            code: Code::SIZE_CAP,
            detail: format!(
                "manifest rows are {input_length} bytes, cap is {}",
                Manifest::MAX_BYTES
            ),
        });
    }
    let bytes = match read(entries_path) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let rows = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
    if rows.is_empty() {
        return bail(Refusal {
            code: Code::MISSING_FIELD,
            detail: "manifest rows list is empty".into(),
        });
    }
    let mut files = Vec::new();
    for line in rows.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            return bail(Refusal {
                code: Code::MALFORMED_JSON,
                detail: "manifest rows contain an empty line".into(),
            });
        }
        let value = match tensorfs_core::canon::parse_canonical(line, Manifest::MAX_BYTES) {
            Ok(value) => value,
            Err(error) => return bail(error),
        };
        let mut fields = match tensorfs_core::canon::Fields::new("ManifestFileRow", &value) {
            Ok(fields) => fields,
            Err(error) => return bail(error),
        };
        let blob = match fields
            .req("blob")
            .and_then(|value| ObjectRef::from_value("ManifestFileRow.blob", value))
        {
            Ok(blob) => blob,
            Err(error) => return bail(error),
        };
        let path = match fields.req_str("path") {
            Ok(path) => path.to_string(),
            Err(error) => return bail(error),
        };
        if let Err(error) = fields.done() {
            return bail(error);
        }
        files.push((path, blob));
    }
    let manifest = match Manifest::from_files(files) {
        Ok(manifest) => manifest,
        Err(error) => return bail(error),
    };
    let manifest_bytes = manifest.canonical_bytes();
    if let Err(error) = write(Path::new(output), &manifest_bytes) {
        return bail(error);
    }
    let reference = ObjectRef::of(&manifest_bytes);
    println!(
        "{}",
        String::from_utf8(tensorfs_core::canon::write(&reference.to_value()))
            .expect("canonical JSON is UTF-8")
    );
    ExitCode::SUCCESS
}

fn read(path: &Path) -> tensorfs_core::err::Result<Vec<u8>> {
    fs::read(path).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {}: {error}", path.display()),
    })
}

fn write(path: &Path, bytes: &[u8]) -> tensorfs_core::err::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("mkdir {}: {error}", parent.display()),
        })?;
    }
    fs::write(path, bytes).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("write {}: {error}", path.display()),
    })
}

fn repo_rows(repository: &Repository, bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut rows = vec![tensorfs_core::canon::write(&Value::obj(vec![
        (
            "document_sha256",
            Value::str(tensorfs_core::sha256::hex_digest(bytes)),
        ),
        ("kind", Value::str("repo")),
        ("name", Value::str(repository.repo.name.clone())),
        ("org", Value::str(repository.repo.org.clone())),
    ]))];
    for release in &repository.releases {
        let mut fields = vec![
            ("kind", Value::str("release")),
            ("lane", Value::str(release.lane.clone())),
            ("manifest_length", Value::uint(release.manifest.length)),
            (
                "manifest_sha256",
                Value::str(release.manifest.sha256.clone()),
            ),
            ("name", Value::str(repository.repo.name.clone())),
            ("org", Value::str(repository.repo.org.clone())),
            ("version", Value::str(release.version.clone())),
        ];
        if let Some(evidence) = &release.evidence {
            fields.push(("evidence", evidence.to_value()));
        }
        if let Some(metadata) = &release.metadata {
            fields.push(("metadata", metadata.to_value()));
        }
        rows.push(tensorfs_core::canon::write(&Value::obj(fields)));
    }
    rows
}

pub fn cmd_repo_inspect(path: &Path, flags: &Flags) -> ExitCode {
    let bytes = match read(path) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let repository = match Repository::parse(&bytes) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let Some(rows) = flag(flags, "rows") else {
        eprintln!("--rows <jsonl> is required");
        return ExitCode::from(2);
    };
    if let Err(error) = storage::write_json_lines(Path::new(rows), &repo_rows(&repository, &bytes))
    {
        return bail(error);
    }
    println!(
        "repository {}/{}: {} releases, sha256:{}",
        repository.repo.org,
        repository.repo.name,
        repository.releases.len(),
        repository.document_sha256()
    );
    ExitCode::SUCCESS
}

pub fn cmd_repo_list(root: &Path, flags: &Flags) -> ExitCode {
    let Some(rows) = flag(flags, "rows") else {
        eprintln!("--rows <jsonl> is required");
        return ExitCode::from(2);
    };
    if let Err(error) = Store::open(root)
        .and_then(|_| storage::repository_release_lines(root))
        .and_then(|lines| storage::write_json_lines(Path::new(rows), &lines))
    {
        return bail(error);
    }
    ExitCode::SUCCESS
}

pub fn cmd_repo_get(root: &Path, org: &str, name: &str, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <current.json> is required");
        return ExitCode::from(2);
    };
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let repository_name = match repository::RepositoryName::new(org, name) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let path = store.repository_path(&repository_name);
    let bytes = match read(&path) {
        Ok(bytes) => bytes,
        Err(mut error) => {
            if !path.is_file() {
                error.code = Code::REPOSITORY_ABSENT;
            }
            return bail(error);
        }
    };
    let repository = match Repository::parse(&bytes) {
        Ok(repository) => repository,
        Err(error) => return bail(error),
    };
    if repository.repo != repository_name {
        return bail(Refusal {
            code: Code::PATH_DIGEST_MISMATCH,
            detail: "repository bytes disagree with requested path".into(),
        });
    }
    if let Err(error) = write(Path::new(output), &bytes) {
        return bail(error);
    }
    println!(
        "repository {org}/{name} sha256:{}",
        repository.document_sha256()
    );
    ExitCode::SUCCESS
}

pub fn cmd_repo_apply(current: &str, mutation: &Path, flags: &Flags) -> ExitCode {
    let (Some(output), Some(rows)) = (flag(flags, "out"), flag(flags, "rows")) else {
        eprintln!("--out <replacement> and --rows <jsonl> are required");
        return ExitCode::from(2);
    };
    let current_bytes = if current == "-" {
        None
    } else {
        match read(Path::new(current)) {
            Ok(value) => Some(value),
            Err(error) => return bail(error),
        }
    };
    let mutation = match read(mutation).and_then(|bytes| Mutation::parse(&bytes)) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };

    let inputs = match &mutation {
        Mutation::PutRelease { manifest, .. } => {
            let (Some(manifest_path), Some(inventory_path)) =
                (flag(flags, "manifest"), flag(flags, "inventory"))
            else {
                eprintln!("put_release requires --manifest <json> --inventory <jsonl>");
                return ExitCode::from(2);
            };
            let manifest_bytes = match read(Path::new(manifest_path)) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            if ObjectRef::of(&manifest_bytes) != *manifest {
                return bail(Refusal {
                    code: Code::OBJECT_ID_MISMATCH,
                    detail: "staged manifest bytes disagree with mutation".into(),
                });
            }
            let parsed = match Manifest::parse(&manifest_bytes) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let (header_bytes, classification) = match (parsed.header(), flag(flags, "header")) {
                (Some(header_ref), Some(header_path)) => {
                    let bytes = match read(Path::new(header_path)) {
                        Ok(value) => value,
                        Err(error) => return bail(error),
                    };
                    if ObjectRef::of(&bytes) != *header_ref {
                        return bail(Refusal {
                            code: Code::OBJECT_ID_MISMATCH,
                            detail: "staged header bytes disagree with manifest".into(),
                        });
                    }
                    let header = match Header::parse(&bytes) {
                        Ok(value) => value,
                        Err(error) => return bail(error),
                    };
                    let topology = header.topology_digest();
                    println!("topology_digest {topology}");
                    (Some(bytes), Some(topology))
                }
                (None, None) => (None, None),
                (Some(_), None) => {
                    eprintln!("a CozyTensors manifest also requires --header <cbor>");
                    return ExitCode::from(2);
                }
                (None, Some(_)) => {
                    eprintln!("--header is invalid for an ordinary-file manifest");
                    return ExitCode::from(2);
                }
            };
            let inventory = match storage::read_inventory(Path::new(inventory_path)) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            Some((manifest_bytes, header_bytes, inventory, classification))
        }
        Mutation::RemoveRelease { .. } | Mutation::DeleteRepository { .. } => None,
    };

    let classification = inputs.as_ref().and_then(|inputs| inputs.3.as_deref());
    let replacement = match repository::apply(current_bytes.as_deref(), &mutation, classification) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let (bytes, lines) = match replacement {
        Some(repository) => {
            let bytes = repository.canonical_bytes();
            let lines = match (&mutation, inputs) {
                (
                    Mutation::PutRelease { lane, version, .. },
                    Some((manifest, header, inventory, _)),
                ) => {
                    let release = repository
                        .releases
                        .iter()
                        .find(|release| &release.lane == lane && &release.version == version)
                        .expect("applied release is present");
                    match Census::project_release_inputs(
                        &repository,
                        &bytes,
                        release,
                        &manifest,
                        header.as_deref(),
                        inventory,
                    ) {
                        Ok(projection) => projection.json_lines(),
                        Err(error) => return bail(error),
                    }
                }
                (Mutation::RemoveRelease { .. }, None) => repo_rows(&repository, &bytes),
                _ => unreachable!("mutation inputs match their action"),
            };
            (bytes, lines)
        }
        None => (Vec::new(), Vec::new()),
    };
    if let Err(error) = write(Path::new(output), &bytes)
        .and_then(|()| storage::write_json_lines(Path::new(rows), &lines))
    {
        return bail(error);
    }
    println!(
        "{}",
        if bytes.is_empty() {
            "delete"
        } else {
            "replace"
        }
    );
    ExitCode::SUCCESS
}

pub fn cmd_repo_commit(store: &Path, current: &str, mutation: &Path, flags: &Flags) -> ExitCode {
    let store = match Store::open(store) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let current = if current == "-" {
        None
    } else {
        match read(Path::new(current)) {
            Ok(value) => Some(value),
            Err(error) => return bail(error),
        }
    };
    let mutation = match read(mutation).and_then(|bytes| Mutation::parse(&bytes)) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let fault = Fault {
        stage: flag(flags, "fault").map(str::to_string),
        ready: flag(flags, "ready").map(std::path::PathBuf::from),
    };
    match store.apply_repository(current.as_deref(), &mutation, &fault) {
        Ok(Some(repository)) => {
            println!("repository sha256:{}", repository.document_sha256());
            ExitCode::SUCCESS
        }
        Ok(None) => {
            println!("repository deleted");
            ExitCode::SUCCESS
        }
        Err(error) => bail(error),
    }
}

pub fn cmd_key(kind: &str, args: &[&str]) -> ExitCode {
    let result = match (kind, args) {
        ("blob", [sha256]) => storage::blob_key(sha256.trim_start_matches("sha256:")),
        ("manifest", [sha256]) => storage::manifest_key(sha256.trim_start_matches("sha256:")),
        ("repo", [org, name]) => tensorfs_core::repository::RepositoryName::new(*org, *name)
            .and_then(|repo| storage::repo_key(&repo)),
        _ => {
            eprintln!("usage: tfs key blob <sha256> | manifest <sha256> | repo <org> <name>");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(key) => {
            println!("{key}");
            ExitCode::SUCCESS
        }
        Err(error) => bail(error),
    }
}

pub fn cmd_manifest_inspect(path: &Path, flags: &Flags) -> ExitCode {
    let bytes = match read(path) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let manifest = match Manifest::parse(&bytes) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    if let Some(expected) = flag(flags, "expect") {
        let (sha256, length) = expected.split_once(':').unwrap_or((expected, ""));
        if sha256.trim_start_matches("sha256:") != tensorfs_core::sha256::hex_digest(&bytes)
            || length.parse::<u64>().ok() != Some(bytes.len() as u64)
        {
            return bail(Refusal {
                code: Code::OBJECT_ID_MISMATCH,
                detail: "manifest --expect differs from observed bytes".into(),
            });
        }
    }
    let Some(refs) = flag(flags, "refs") else {
        eprintln!("--refs <jsonl> is required");
        return ExitCode::from(2);
    };
    let direct: Vec<Vec<u8>> = manifest
        .entries()
        .iter()
        .map(|(path, entry)| {
            tensorfs_core::canon::write(&Value::obj(vec![
                (
                    "kind",
                    Value::str(match entry {
                        Entry::File(_) => "file",
                        Entry::CozyTensors(_) => "cozytensors",
                    }),
                ),
                ("length", Value::uint(entry.blob().length)),
                ("path", Value::str(path.clone())),
                ("sha256", Value::str(entry.blob().sha256.clone())),
            ]))
        })
        .collect();
    if let Err(error) = storage::write_json_lines(Path::new(refs), &direct) {
        return bail(error);
    }
    match (
        flag(flags, "header"),
        flag(flags, "inventory"),
        flag(flags, "closure"),
    ) {
        (None, None, None) => {}
        (Some(header), Some(inventory), Some(closure)) => {
            let header_ref = match manifest.header() {
                Some(value) => value,
                None => {
                    return bail(Refusal {
                        code: Code::ATTACHMENT_CARDINALITY,
                        detail: "manifest has no cozytensors entry".into(),
                    })
                }
            };
            let header_bytes = match read(Path::new(header)) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            if ObjectRef::of(&header_bytes) != *header_ref {
                return bail(Refusal {
                    code: Code::OBJECT_ID_MISMATCH,
                    detail: "--header bytes differ from the manifest's cozytensors ref".into(),
                });
            }
            let header = match Header::parse(&header_bytes) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let inventory = match storage::read_inventory(Path::new(inventory)) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let refs = match closure_refs(&manifest, &header, header_ref) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            for (_, object) in refs.values() {
                if inventory.get(&object.sha256).map(|entry| entry.length) != Some(object.length) {
                    return bail(Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail: format!(
                            "{} is absent or length-mismatched in inventory",
                            object.id()
                        ),
                    });
                }
            }
            let lines: Vec<Vec<u8>> = refs
                .values()
                .map(|(kind, object)| {
                    tensorfs_core::canon::write(&Value::obj(vec![
                        ("length", Value::uint(object.length)),
                        ("object_kind", Value::str(*kind)),
                        ("sha256", Value::str(object.sha256.clone())),
                    ]))
                })
                .collect();
            if let Err(error) = storage::write_json_lines(Path::new(closure), &lines) {
                return bail(error);
            }
        }
        _ => {
            eprintln!("--header, --inventory and --closure must be supplied together");
            return ExitCode::from(2);
        }
    }
    println!(
        "manifest sha256:{} length={} entries={}",
        tensorfs_core::sha256::hex_digest(&bytes),
        bytes.len(),
        manifest.entries().len()
    );
    ExitCode::SUCCESS
}

fn closure_refs(
    manifest: &Manifest,
    header: &Header,
    header_ref: &ObjectRef,
) -> tensorfs_core::err::Result<BTreeMap<String, (&'static str, ObjectRef)>> {
    let mut refs = BTreeMap::new();
    insert(&mut refs, "header", header_ref)?;
    for (_, entry) in manifest.entries() {
        if let Entry::File(object) = entry {
            insert(&mut refs, "file", object)?;
        }
    }
    for (_, config) in &header.configs {
        insert(
            &mut refs,
            "config",
            &ObjectRef {
                sha256: config.sha256.clone(),
                length: config.length,
            },
        )?;
    }
    for (_, asset) in &header.assets {
        for object in &asset.segments {
            insert(&mut refs, "model_asset", object)?;
        }
    }
    for (_, _, tensor) in header.tensors() {
        for (_, part) in &tensor.parts {
            if let Body::Segments(segments) = &part.body {
                for object in segments {
                    insert(&mut refs, "part", object)?;
                }
            }
        }
    }
    Ok(refs)
}

fn insert(
    refs: &mut BTreeMap<String, (&'static str, ObjectRef)>,
    kind: &'static str,
    object: &ObjectRef,
) -> tensorfs_core::err::Result<()> {
    if let Some((_, existing)) = refs.get(&object.sha256) {
        if existing.length != object.length {
            return Err(Refusal {
                code: Code::LENGTH_MISMATCH,
                detail: format!("{} appears at conflicting lengths", object.id()),
            });
        }
    } else {
        refs.insert(object.sha256.clone(), (kind, object.clone()));
    }
    Ok(())
}

pub fn cmd_gc_headers(census: &Path, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <requests.jsonl> is required");
        return ExitCode::from(2);
    };
    let census = match Census::open_metadata(census) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let requests = match census.header_requests() {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let lines: Vec<Vec<u8>> = requests.iter().map(InventoryEntry::line).collect();
    match storage::write_json_lines(Path::new(output), &lines) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => bail(error),
    }
}

pub fn cmd_gc_holds(store: &Path, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <holds.jsonl> is required");
        return ExitCode::from(2);
    };
    let catalog = match Store::open(store).and_then(|_| Catalog::open(store)) {
        Ok(catalog) => catalog,
        Err(error) => return bail(error),
    };
    let holds = match catalog.held_keys() {
        Ok(holds) => holds,
        Err(error) => return bail(error),
    };
    let lines: Vec<Vec<u8>> = holds
        .iter()
        .map(|hold| {
            tensorfs_core::canon::write(&Value::obj(vec![
                ("key", Value::str(hold.key.clone())),
                ("kind", Value::str(hold.kind.clone())),
                ("length", Value::uint(hold.length)),
            ]))
        })
        .collect();
    match storage::write_json_lines(Path::new(output), &lines) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => bail(error),
    }
}

pub fn cmd_gc_plan(census: &Path, flags: &Flags) -> ExitCode {
    let (Some(holds), Some(output)) = (flag(flags, "holds"), flag(flags, "out")) else {
        eprintln!("--holds <holds.jsonl> and --out <plan.jsonl> are required");
        return ExitCode::from(2);
    };
    let census = match Census::open(census) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let holds = match storage::read_holds(Path::new(holds)) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let plan = match census.gc_plan(&holds) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let lines: Vec<Vec<u8>> = plan.iter().map(|row| row.line()).collect();
    match storage::write_json_lines(Path::new(output), &lines) {
        Ok(()) => {
            println!("{} unreferenced keys", plan.len());
            ExitCode::SUCCESS
        }
        Err(error) => bail(error),
    }
}

pub fn cmd_rebuild(census: &Path, flags: &Flags) -> ExitCode {
    if let Err(error) = WriterGuard::refuse_rebuild_if_live(census) {
        return bail(error);
    }
    let census_value = match Census::open(census) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let projection = match census_value.projection() {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    if let Some(rows) = flag(flags, "rows") {
        if let Err(error) = storage::write_json_lines(Path::new(rows), &projection.json_lines()) {
            return bail(error);
        }
    }
    if let Some(sqlite) = flag(flags, "sqlite") {
        let path = Path::new(sqlite);
        if path.file_name().and_then(|value| value.to_str()) != Some("tensorfs.sqlite") {
            eprintln!("--sqlite path must end in tensorfs.sqlite");
            return ExitCode::from(2);
        }
        let Some(root) = path.parent() else {
            return ExitCode::from(2);
        };
        let catalog = match Catalog::initialize(root) {
            Ok(value) => value,
            Err(error) => return bail(error),
        };
        if let Err(error) = catalog.replace_all(&projection) {
            return bail(error);
        }
    }
    println!(
        "rebuilt {} repos, {} releases, {} release objects",
        projection.repos.len(),
        projection.releases.len(),
        projection.objects.len()
    );
    ExitCode::SUCCESS
}
