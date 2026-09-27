use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::canon::Value;
use tensorfs_core::catalog::{Catalog, WriterGuard};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::gc;
use tensorfs_core::header::{Body, Header};
use tensorfs_core::ids::{hex64, prefixed, Doc, ObjectRef};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::reclaim;
use tensorfs_core::repository::{self, Mutation, Repository};
use tensorfs_core::storage::{
    self, Census, CheckpointFacts as CensusCheckpointFacts, InventoryEntry, StagedCheckpoint,
};
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
    for checkpoint in &repository.checkpoints {
        let fields = vec![
            ("kind", Value::str("checkpoint")),
            ("manifest_length", Value::uint(checkpoint.manifest.length)),
            (
                "manifest_sha256",
                Value::str(checkpoint.manifest.sha256.clone()),
            ),
            ("name", Value::str(repository.repo.name.clone())),
            ("org", Value::str(repository.repo.org.clone())),
            ("published", Value::Bool(checkpoint.published)),
        ];
        rows.push(tensorfs_core::canon::write(&Value::obj(fields)));
    }
    for release in &repository.releases {
        for lane in &release.lanes {
            let fields = vec![
                ("kind", Value::str("release")),
                ("lane", Value::str(lane.lane.clone())),
                ("manifest_length", Value::uint(lane.manifest.length)),
                ("manifest_sha256", Value::str(lane.manifest.sha256.clone())),
                ("name", Value::str(repository.repo.name.clone())),
                ("org", Value::str(repository.repo.org.clone())),
                ("release_revision", Value::uint(release.revision)),
                ("release_yanked", Value::Bool(release.yanked)),
                ("version", Value::str(release.version.clone())),
            ];
            rows.push(tensorfs_core::canon::write(&Value::obj(fields)));
        }
    }
    rows
}

fn checkpoint_fact_rows(
    repository: &Repository,
    manifest: &ObjectRef,
    facts: &CensusCheckpointFacts,
) -> Vec<Vec<u8>> {
    let mut fields = vec![
        ("kind", Value::str("checkpoint_facts")),
        ("manifest_length", Value::uint(manifest.length)),
        ("manifest_sha256", Value::str(manifest.sha256.clone())),
        ("name", Value::str(repository.repo.name.clone())),
        ("object_bytes", Value::uint(facts.object_bytes)),
        ("object_count", Value::uint(facts.object_count)),
        ("org", Value::str(repository.repo.org.clone())),
    ];
    if let Some(header) = &facts.header {
        fields.push(("header_length", Value::uint(header.length)));
        fields.push(("header_sha256", Value::str(header.sha256.clone())));
    }
    let mut rows = vec![tensorfs_core::canon::write(&Value::obj(fields))];
    for (object_kind, object) in &facts.objects {
        rows.push(tensorfs_core::canon::write(&Value::obj(vec![
            ("kind", Value::str("checkpoint_object")),
            ("length", Value::uint(object.length)),
            ("manifest_sha256", Value::str(manifest.sha256.clone())),
            ("name", Value::str(repository.repo.name.clone())),
            ("object_kind", Value::str(object_kind.clone())),
            ("org", Value::str(repository.repo.org.clone())),
            ("sha256", Value::str(object.sha256.clone())),
        ])));
    }
    for (encoding_id, alias) in &facts.encodings {
        let mut fields = vec![
            ("encoding_id", Value::str(encoding_id.clone())),
            ("kind", Value::str("checkpoint_encoding")),
            ("manifest_sha256", Value::str(manifest.sha256.clone())),
            ("name", Value::str(repository.repo.name.clone())),
            ("org", Value::str(repository.repo.org.clone())),
        ];
        if let Some(alias) = alias {
            fields.push(("alias", Value::str(alias.clone())));
        }
        rows.push(tensorfs_core::canon::write(&Value::obj(fields)));
    }
    for component in &facts.components {
        rows.push(tensorfs_core::canon::write(&Value::obj(vec![
            ("component", Value::str(component.clone())),
            ("kind", Value::str("checkpoint_component")),
            ("manifest_sha256", Value::str(manifest.sha256.clone())),
            ("name", Value::str(repository.repo.name.clone())),
            ("org", Value::str(repository.repo.org.clone())),
        ])));
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
pub fn cmd_repo_usage(root: &Path, flags: &Flags) -> ExitCode {
    let Some(rows) = flag(flags, "rows") else {
        eprintln!("--rows <jsonl> is required");
        return ExitCode::from(2);
    };
    let usage = match Store::open(root)
        .and_then(|_| Census::open(root))
        .and_then(|census| census.usage())
    {
        Ok(usage) => usage,
        Err(error) => return bail(error),
    };
    if let Err(error) = storage::write_json_lines(Path::new(rows), &usage.json_lines()) {
        return bail(error);
    }
    println!(
        "{} repos: {} bytes referenced, {} bytes unique, {} bytes unreferenced",
        usage.repos.len(),
        usage.bytes_total,
        usage.bytes_unique_sum,
        usage.bytes_unreferenced
    );
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

fn local_row(repository: &Repository) -> Result<Value, Refusal> {
    let release = repository.local_checkpoint()?;
    let row = Value::obj(vec![
        ("manifest_digest", Value::str(release.manifest.id())),
        ("manifest_length", Value::uint(release.manifest.length)),
        ("name", Value::str(repository.repo.name.clone())),
        (
            "repository_digest",
            Value::str(format!("sha256:{}", repository.document_sha256())),
        ),
        (
            "source_selection",
            Value::str(format!(
                "sha256:{}",
                release
                    .source_selection
                    .as_ref()
                    .expect("validated local checkpoint")
            )),
        ),
    ]);
    Ok(row)
}

fn print_local(repository: &Repository) -> Result<(), Refusal> {
    let row = local_row(repository)?;
    println!(
        "{}",
        String::from_utf8(tensorfs_core::canon::write(&row)).unwrap()
    );
    Ok(())
}

fn observed_local(store: &Store, name: &str, observed: &str) -> Result<Option<Vec<u8>>, Refusal> {
    if observed == "absent" {
        return Ok(None);
    }
    let expected = prefixed("--observed", observed)?;
    let repo = repository::RepositoryName::new("local", name)?;
    let path = store.repository_path(&repo);
    let bytes = read(&path)?;
    let parsed = Repository::parse(&bytes)?;
    local_row(&parsed)?;
    let actual = format!("sha256:{}", parsed.document_sha256());
    if actual != expected {
        return Err(Refusal {
            code: Code::REPOSITORY_CONFLICT,
            detail: format!("observed local repository {expected}, current is {actual}"),
        });
    }
    Ok(Some(bytes))
}

pub fn cmd_local_resolve(root: &Path, name: &str) -> ExitCode {
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let repo = match repository::RepositoryName::new("local", name) {
        Ok(repo) => repo,
        Err(error) => return bail(error),
    };
    let path = store.repository_path(&repo);
    let bytes = match read(&path) {
        Ok(bytes) => bytes,
        Err(mut error) if !path.is_file() => {
            error.code = Code::REPOSITORY_ABSENT;
            return bail(error);
        }
        Err(error) => return bail(error),
    };
    match Repository::parse(&bytes).and_then(|repository| print_local(&repository)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => bail(error),
    }
}

pub fn cmd_local_replace(
    root: &Path,
    name: &str,
    source_selection: &str,
    manifest_id: &str,
    manifest_length: &str,
    flags: &Flags,
) -> ExitCode {
    let Some(observed) = flag(flags, "observed") else {
        eprintln!("--observed <sha256:...|absent> is required");
        return ExitCode::from(2);
    };
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let observed = match observed_local(&store, name, observed) {
        Ok(observed) => observed,
        Err(error) => return bail(error),
    };
    let selection = match hex64(
        "source selection",
        source_selection.trim_start_matches("sha256:"),
    ) {
        Ok(selection) => selection,
        Err(error) => return bail(error),
    };
    let length = match manifest_length.parse::<u64>() {
        Ok(length) if length > 0 => length,
        _ => {
            return bail(Refusal {
                code: Code::LENGTH_MISMATCH,
                detail: "manifest length must be a positive integer".into(),
            })
        }
    };
    let manifest = ObjectRef {
        sha256: manifest_id.trim_start_matches("sha256:").to_string(),
        length,
    };
    if let Err(error) = hex64("manifest", &manifest.sha256) {
        return bail(error);
    }
    let mutation = Mutation::ReplaceLocal {
        repo: match repository::RepositoryName::new("local", name) {
            Ok(repo) => repo,
            Err(error) => return bail(error),
        },
        manifest,
        version: selection,
    };
    match store.apply_repository(observed.as_deref(), &mutation, &Fault::default()) {
        Ok(Some(repository)) => match print_local(&repository) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => bail(error),
        },
        Ok(None) => unreachable!("replace_local retains the repository"),
        Err(error) => bail(error),
    }
}

pub fn cmd_local_remove(root: &Path, name: &str, flags: &Flags) -> ExitCode {
    let Some(observed) = flag(flags, "observed") else {
        eprintln!("--observed <sha256:...> is required");
        return ExitCode::from(2);
    };
    if observed == "absent" {
        eprintln!("--observed absent is invalid for local remove");
        return ExitCode::from(2);
    }
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let observed = match observed_local(&store, name, observed) {
        Ok(Some(observed)) => observed,
        Ok(None) => unreachable!("absent was rejected above"),
        Err(error) => return bail(error),
    };
    let mutation = Mutation::DeleteRepository {
        repo: match repository::RepositoryName::new("local", name) {
            Ok(repo) => repo,
            Err(error) => return bail(error),
        },
    };
    match store.apply_repository(Some(&observed), &mutation, &Fault::default()) {
        Ok(None) => {
            let row = Value::obj(vec![
                ("name", Value::str(name.to_string())),
                ("removed", Value::Bool(true)),
            ]);
            println!(
                "{}",
                String::from_utf8(tensorfs_core::canon::write(&row)).unwrap()
            );
            ExitCode::SUCCESS
        }
        Ok(Some(_)) => unreachable!("delete repository returns none"),
        Err(error) => bail(error),
    }
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
        Mutation::PutCheckpoint { manifest, .. } | Mutation::ReplaceLocal { manifest, .. } => {
            let (Some(manifest_path), Some(inventory_path)) =
                (flag(flags, "manifest"), flag(flags, "inventory"))
            else {
                eprintln!("manifest mutation requires --manifest <json> --inventory <jsonl>");
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
            let header_bytes = match (parsed.header(), flag(flags, "header")) {
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
                    match Header::parse(&bytes) {
                        Ok(_) => {}
                        Err(error) => return bail(error),
                    }
                    Some(bytes)
                }
                (None, None) => None,
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
            Some((manifest_bytes, header_bytes, inventory))
        }
        Mutation::RemoveCheckpoint { .. }
        | Mutation::UpdateRelease { .. }
        | Mutation::YankRelease { .. }
        | Mutation::DeleteRepository { .. } => None,
    };

    let replacement = match repository::apply(current_bytes.as_deref(), &mutation) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let (bytes, lines) = match replacement {
        Some(repository) => {
            let bytes = repository.canonical_bytes();
            let lines = match (&mutation, inputs) {
                (
                    Mutation::PutCheckpoint { manifest, .. },
                    Some((manifest_bytes, header, inventory)),
                ) => {
                    let facts = match Census::validate_checkpoint_inputs(
                        manifest,
                        StagedCheckpoint {
                            manifest: &manifest_bytes,
                            header: header.as_deref(),
                        },
                        &inventory,
                    ) {
                        Ok(facts) => facts,
                        Err(error) => return bail(error),
                    };
                    let mut rows = repo_rows(&repository, &bytes);
                    rows.extend(checkpoint_fact_rows(&repository, manifest, &facts));
                    rows
                }
                (
                    Mutation::ReplaceLocal { manifest, .. },
                    Some((manifest_bytes, header, inventory)),
                ) => {
                    let facts = match Census::validate_checkpoint_inputs(
                        manifest,
                        StagedCheckpoint {
                            manifest: &manifest_bytes,
                            header: header.as_deref(),
                        },
                        &inventory,
                    ) {
                        Ok(facts) => facts,
                        Err(error) => return bail(error),
                    };
                    let mut rows = repo_rows(&repository, &bytes);
                    rows.extend(checkpoint_fact_rows(&repository, manifest, &facts));
                    rows
                }
                (Mutation::RemoveCheckpoint { .. }, None) => repo_rows(&repository, &bytes),
                (Mutation::UpdateRelease { .. }, None) => repo_rows(&repository, &bytes),
                (Mutation::YankRelease { .. }, None) => repo_rows(&repository, &bytes),
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
                ("kind", Value::str(entry.kind())),
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
    // Runtime roles precede ordinary snapshot paths so an aliased README/sample path cannot
    // relabel a required config, model asset, or tensor part as an ignorable file.
    for (_, entry) in manifest.entries() {
        if let Some(object) = entry.content() {
            insert(&mut refs, "file", object)?;
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

/// The store's holds as the filesystem states them: every candidate an open ingest session
/// root names (`tmp/ingest/<session>/session.json`). The catalog's hold rows are not read —
/// the database is not the source of truth (tensorfs_core::gc).
pub fn cmd_gc_holds(store: &Path, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <holds.jsonl> is required");
        return ExitCode::from(2);
    };
    let (holds, _) = match Store::open(store).and_then(|store| gc::session_holds(&store)) {
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
    let _gc = match WriterGuard::lock_rebuild(census) {
        Ok(guard) => guard,
        Err(error) => return bail(error),
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

/// `tfs gc <root> [--dry-run] [--json]`: one complete reclamation pass (tensorfs_core::gc).
pub fn cmd_gc(root: &Path, flags: &Flags) -> ExitCode {
    let report = match gc::collect(root, flag(flags, "dry-run").is_some()) {
        Ok(report) => report,
        Err(error) => return bail(error),
    };
    if flag(flags, "json").is_some() {
        println!("{}", String::from_utf8_lossy(&report.json()));
    } else {
        println!("{}", report.line());
        for session in &report.sessions {
            println!("held by ingest session {session} (abandon with `tfs ingest reap`)");
        }
    }
    ExitCode::SUCCESS
}

pub fn cmd_rebuild(census: &Path, flags: &Flags) -> ExitCode {
    let _rebuild = match WriterGuard::lock_rebuild(census) {
        Ok(guard) => guard,
        Err(error) => return bail(error),
    };
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
        "rebuilt {} repos, {} releases, {} checkpoint objects",
        projection.repos.len(),
        projection.releases.len(),
        projection.objects.len()
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------- reclamation (tfs-065)
//
// `gc` collects what nothing names; `reclaim` chooses what to stop naming. Two acts, two
// names — and this half only ever produces a document.

/// The holds a plan must respect: the same ones `tfs gc` computes, so a reclaim estimate
/// can never promise bytes an open ingest session is standing on.
fn reclaim_holds(store: &Store) -> Result<Vec<storage::HeldKey>, Refusal> {
    gc::session_holds(store).map(|(holds, _)| holds)
}

/// Read a `{"name":..,"org":..}` JSONL list of repositories. The same shape `tfs repo
/// list --rows` emits, so a caller filters that output rather than inventing a format.
fn read_repo_list(path: &str) -> Result<Vec<repository::RepositoryName>, Refusal> {
    let bytes = fs::read(Path::new(path)).map_err(|error| Refusal {
        code: Code::IO_FAILED,
        detail: format!("read {path}: {error}"),
    })?;
    let mut names = Vec::new();
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let value = tensorfs_core::canon::parse_canonical(line, 16 * 1024)?;
        let mut fields = tensorfs_core::canon::Fields::new("RepositoryRow", &value)?;
        let name = fields.req_str("name")?.to_string();
        let org = fields.req_str("org")?.to_string();
        names.push(repository::RepositoryName::new(org, name)?);
    }
    Ok(names)
}

pub fn cmd_reclaim_usage(root: &Path, flags: &Flags) -> ExitCode {
    let Some(rows) = flag(flags, "rows") else {
        eprintln!("--rows <jsonl> is required");
        return ExitCode::from(2);
    };
    let mut keep = Vec::new();
    let mut candidates = Vec::new();
    for (list, into) in [("keep", &mut keep), ("candidates", &mut candidates)] {
        if let Some(path) = flag(flags, list) {
            match read_repo_list(path) {
                Ok(names) => *into = names,
                Err(error) => return bail(error),
            }
        }
    }
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let usage = match reclaim_holds(&store)
        .and_then(|holds| reclaim::usage(&store, &keep, &candidates, &holds))
    {
        Ok(usage) => usage,
        Err(error) => return bail(error),
    };
    if let Err(error) = storage::write_json_lines(Path::new(rows), &usage.json_lines()) {
        return bail(error);
    }
    // Both per-root numbers, always. `bytes_closure` alone reads as "what this model costs"
    // and is the figure that picks the wrong victim; `bytes_marginal` alone hides that
    // dropping two roots together frees what neither frees apart — which is why the set
    // total is printed beside them rather than left to be summed.
    for root in &usage.roots {
        println!(
            "{}/{}  closure {} B  marginal {} B",
            root.org, root.name, root.bytes_closure, root.bytes_solo
        );
    }
    println!(
        "all {} candidate(s) together free {} B ({} B already unreferenced); holding {} B \
         of a {} B budget, {} B of headroom",
        usage.roots.len(),
        usage.bytes_total,
        usage.bytes_unreferenced,
        usage.occupancy_bytes,
        usage.budget_bytes,
        usage.headroom_bytes,
    );
    ExitCode::SUCCESS
}

pub fn cmd_reclaim_plan(root: &Path, flags: &Flags) -> ExitCode {
    let Some(need) = flag(flags, "need") else {
        eprintln!("--need <bytes> is required");
        return ExitCode::from(2);
    };
    let Ok(need) = need.parse::<u64>() else {
        eprintln!("--need takes a byte count, not {need:?}");
        return ExitCode::from(2);
    };
    let mut keep = Vec::new();
    for (name, value) in flags {
        if name != "keep" {
            continue;
        }
        let Some((org, repo)) = value.split_once('/') else {
            eprintln!("--keep takes <org>/<name>, not {value:?}");
            return ExitCode::from(2);
        };
        match repository::RepositoryName::new(org, repo) {
            Ok(value) => keep.push(value),
            Err(error) => return bail(error),
        }
    }
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let plan =
        match reclaim_holds(&store).and_then(|holds| reclaim::plan(&store, need, &keep, &holds)) {
            Ok(plan) => plan,
            Err(error) => return bail(error),
        };
    if let Some(output) = flag(flags, "out") {
        if let Err(error) = fs::write(Path::new(output), plan.json()) {
            return bail(Refusal {
                code: Code::IO_FAILED,
                detail: format!("write {output}: {error}"),
            });
        }
    }
    if crate::flag_on(flags, "json") {
        println!("{}", String::from_utf8_lossy(&plan.json()));
    } else {
        println!("{}", plan.line());
        for victim in &plan.victims {
            println!(
                "  drop {}/{}  frees {} B (closure {} B)  cumulative {} B",
                victim.org,
                victim.name,
                victim.marginal_bytes,
                victim.bytes_closure,
                victim.cumulative_bytes
            );
        }
    }
    ExitCode::SUCCESS
}

pub fn cmd_reclaim_run(root: &Path, flags: &Flags) -> ExitCode {
    let Some(plan) = flag(flags, "plan") else {
        eprintln!("--plan <plan.json> is required");
        return ExitCode::from(2);
    };
    let document = match fs::read(Path::new(plan)) {
        Ok(bytes) => bytes,
        Err(error) => {
            return bail(Refusal {
                code: Code::IO_FAILED,
                detail: format!("read {plan}: {error}"),
            })
        }
    };
    let drop = match reclaim::victims_of(&document) {
        Ok(names) => names,
        Err(error) => return bail(error),
    };
    let options = reclaim::RunOptions {
        demote: crate::flag_on(flags, "demote"),
        assume_upstream: crate::flag_on(flags, "assume-upstream"),
    };
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let report = match reclaim::run(&store, &drop, options) {
        Ok(report) => report,
        Err(error) => return bail(error),
    };
    if crate::flag_on(flags, "json") {
        println!("{}", String::from_utf8_lossy(&report.json()));
    } else {
        println!("{}", report.line());
        for dropped in &report.roots_dropped {
            println!("  dropped {dropped}");
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tensorfs_core::manifest::{Draft, Entry};

    #[test]
    fn inspect_closure_preserves_runtime_roles_over_aliased_files() {
        let vectors = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vectors/cases");
        let header = Header::parse(
            &fs::read(vectors.join("header/plain-fp8.cbor")).expect("read frozen header"),
        )
        .unwrap();
        let manifest = Manifest::parse(
            &fs::read(vectors.join("manifest/checkpoint-extra-file.json"))
                .expect("read frozen manifest"),
        )
        .unwrap();
        let header_ref = manifest.header().unwrap().clone();
        let part = header
            .tensors()
            .find_map(|(_, _, tensor)| {
                tensor
                    .parts
                    .iter()
                    .find_map(|(_, part)| part.segments().first().cloned())
            })
            .expect("frozen header has a segmented part");
        let asset = header
            .assets
            .iter()
            .find_map(|(_, asset)| asset.segments.first().cloned())
            .expect("frozen header has a segmented asset");
        let mut draft = Draft::of(&manifest);
        draft.entries.extend([
            ("assets/aliased-model.bin".into(), Entry::File(asset)),
            ("weights/aliased-part.bin".into(), Entry::File(part.clone())),
        ]);
        draft.entries.sort_by(|left, right| left.0.cmp(&right.0));
        let manifest = draft.seal().unwrap();
        let refs = closure_refs(&manifest, &header, &header_ref).unwrap();
        let file_ids: HashSet<&str> = manifest
            .entries()
            .iter()
            .filter_map(|(_, entry)| entry.content().map(|object| object.sha256.as_str()))
            .collect();

        assert!(header.assets.iter().any(|(_, asset)| asset
            .segments
            .iter()
            .any(|segment| file_ids.contains(segment.sha256.as_str()))));
        for (_, asset) in &header.assets {
            for segment in &asset.segments {
                assert_eq!(refs[&segment.sha256].0, "model_asset");
            }
        }
        assert_eq!(refs[&part.sha256].0, "part");

        let runtime_ids: HashSet<&str> = refs
            .values()
            .filter(|(kind, _)| *kind != "file")
            .map(|(_, object)| object.sha256.as_str())
            .collect();
        assert!(manifest
            .entries()
            .iter()
            .any(|(_, entry)| entry.content().is_some_and(|object| {
                !runtime_ids.contains(object.sha256.as_str()) && refs[&object.sha256].0 == "file"
            })));
    }
}
