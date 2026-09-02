use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::canon::Value;
use tensorfs_core::catalog::{Catalog, WriterGuard};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::header::{Body, Header};
use tensorfs_core::ids::{hex64, prefixed, Doc, ObjectRef};
use tensorfs_core::manifest::{Entry, Manifest};
use tensorfs_core::repository::{self, Mutation, Repository};
use tensorfs_core::storage::{
    self, Census, CheckpointFacts as CensusCheckpointFacts, InventoryEntry, StagedCheckpoint,
};
use tensorfs_core::store::{ExpectedCheckpointContract, Fault, Store};

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
    for checkpoint in &repository.checkpoints {
        let mut fields = vec![
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
        if let Some(evidence) = &checkpoint.evidence {
            fields.push(("evidence", evidence.to_value()));
        }
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
    if let (Some(header), Some(digests)) = (&facts.header, &facts.digests) {
        if let Some(config) = &digests.config_digest {
            fields.push(("config_digest", Value::str(config.clone())));
        }
        fields.push(("header_length", Value::uint(header.length)));
        fields.push(("header_sha256", Value::str(header.sha256.clone())));
        fields.push(("shape_digest", Value::str(digests.shape_digest.clone())));
        fields.push((
            "topology_digest",
            Value::str(digests.topology_digest.clone()),
        ));
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
    for (config, sha256, length, media_type) in &facts.configs {
        rows.push(tensorfs_core::canon::write(&Value::obj(vec![
            ("config", Value::str(config.clone())),
            ("kind", Value::str("checkpoint_config")),
            ("length", Value::uint(*length)),
            ("manifest_sha256", Value::str(manifest.sha256.clone())),
            ("media_type", Value::str(media_type.clone())),
            ("name", Value::str(repository.repo.name.clone())),
            ("org", Value::str(repository.repo.org.clone())),
            ("sha256", Value::str(sha256.clone())),
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

/// `tfs repo migrate <store>`: re-derive every checkpoint row's `topology_digest`,
/// `shape_digest` and `config_digest` from the header its manifest names, and rewrite
/// the repository document when the row disagrees or predates the three-digest row
/// (tfs-042 — such a row carried one `classification_digest`). Reads the documents
/// loosely on purpose: a row this verb exists to fix does not parse strictly. Follow with
/// `tfs store rebuild <store> --sqlite <store>/tensorfs.sqlite`.
pub fn cmd_repo_migrate(root: &Path, flags: &Flags) -> ExitCode {
    let store = match Store::open(root) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let _rebuild = match WriterGuard::lock_rebuild(root) {
        Ok(guard) => guard,
        Err(error) => return bail(error),
    };
    let dry = flag(flags, "dry-run").is_some();
    let mut paths = Vec::new();
    for org in fs::read_dir(root.join("repos"))
        .into_iter()
        .flatten()
        .flatten()
    {
        for entry in fs::read_dir(org.path()).into_iter().flatten().flatten() {
            if entry.path().extension().and_then(|value| value.to_str()) == Some("json") {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();
    let (mut rewritten, mut rows) = (0usize, 0usize);
    for path in paths {
        let bytes = match read(&path) {
            Ok(value) => value,
            Err(error) => return bail(error),
        };
        let mut document = match tensorfs_core::canon::parse(
            &bytes,
            tensorfs_core::repository::MAX_REPOSITORY_BYTES,
        ) {
            Ok(value) => value,
            Err(error) => return bail(error),
        };
        let Value::Obj(fields) = &mut document else {
            eprintln!("REFUSED WRONG_TYPE: {} is not an object", path.display());
            return ExitCode::FAILURE;
        };
        let Some((_, Value::Arr(checkpoints))) =
            fields.iter_mut().find(|(k, _)| k == "checkpoints")
        else {
            eprintln!(
                "REFUSED MISSING_FIELD: {} has no checkpoints",
                path.display()
            );
            return ExitCode::FAILURE;
        };
        for checkpoint in checkpoints.iter_mut() {
            let Value::Obj(row) = checkpoint else {
                continue;
            };
            let manifest = match row.iter().find(|(k, _)| k == "manifest") {
                Some((_, value)) => match ObjectRef::from_value("checkpoint.manifest", value) {
                    Ok(value) => value,
                    Err(error) => return bail(error),
                },
                None => continue,
            };
            let Some((_, Value::Obj(evidence))) = row.iter_mut().find(|(k, _)| k == "evidence")
            else {
                continue;
            };
            let manifest = match store.read_manifest(&manifest) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let Some(header_ref) = manifest.header() else {
                continue;
            };
            let header = match tensorfs_core::checkpoint::load_header(&store, header_ref) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            let digests = match tensorfs_core::checkpoint::digests(&store, &header) {
                Ok(value) => value,
                Err(error) => return bail(error),
            };
            evidence.retain(|(k, _)| {
                !matches!(
                    k.as_str(),
                    "classification_digest" | "topology_digest" | "shape_digest" | "config_digest"
                )
            });
            if let Some(config) = digests.config_digest {
                evidence.push(("config_digest".to_string(), Value::str(config)));
            }
            evidence.push(("shape_digest".to_string(), Value::str(digests.shape_digest)));
            evidence.push((
                "topology_digest".to_string(),
                Value::str(digests.topology_digest),
            ));
            evidence.sort_by(|a, b| a.0.cmp(&b.0));
            rows += 1;
        }
        let replacement = match Repository::parse(&tensorfs_core::canon::write(&document)) {
            Ok(value) => value.canonical_bytes(),
            Err(error) => return bail(error),
        };
        if replacement == bytes {
            continue;
        }
        rewritten += 1;
        println!("migrate      {}", path.display());
        if dry {
            continue;
        }
        let temporary = path.with_extension(format!("json.migrate.{}", std::process::id()));
        if let Err(error) = write(&temporary, &replacement).and_then(|()| {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600)).map_err(
                |error| Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("chmod {}: {error}", temporary.display()),
                },
            )?;
            fs::rename(&temporary, &path).map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: format!("rename {}: {error}", path.display()),
            })
        }) {
            let _ = fs::remove_file(&temporary);
            return bail(error);
        }
    }
    println!(
        "{rows} checkpoint row(s) re-derived, {rewritten} repository document(s) {}",
        if dry { "would change" } else { "rewritten" }
    );
    if rewritten > 0 && !dry {
        println!(
            "next: tfs store rebuild {} --sqlite {}",
            root.display(),
            root.join("tensorfs.sqlite").display()
        );
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
    let (Some(observed), Some(evidence_path)) = (flag(flags, "observed"), flag(flags, "evidence"))
    else {
        eprintln!("--observed <sha256:...|absent> and --evidence <evidence.json> are required");
        return ExitCode::from(2);
    };
    let store = match Store::open(root) {
        Ok(store) => store,
        Err(error) => return bail(error),
    };
    let topologies: Vec<&str> = flags
        .iter()
        .filter(|(name, _)| name == "expected-topology")
        .map(|(_, value)| value.as_str())
        .collect();
    if topologies.len() > 1 {
        return bail(Refusal {
            code: Code::ATTACHMENT_CARDINALITY,
            detail: "--expected-topology must appear at most once".into(),
        });
    }
    let encoding_ids: Vec<String> = flags
        .iter()
        .filter(|(name, _)| name == "expected-encoding")
        .map(|(_, value)| value.clone())
        .collect();
    let expected = match ExpectedCheckpointContract::optional(
        topologies.first().copied(),
        (!encoding_ids.is_empty()).then_some(encoding_ids),
    ) {
        Ok(expected) => expected,
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
    let evidence = match read(Path::new(evidence_path))
        .and_then(|bytes| tensorfs_core::repository::CheckpointEvidence::parse(&bytes))
    {
        Ok(evidence) => evidence,
        Err(error) => return bail(error),
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
        evidence: Box::new(evidence),
    };
    match store.apply_repository_with_expected_local(
        observed.as_deref(),
        &mutation,
        expected.as_ref(),
        &Fault::default(),
    ) {
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

/// The exact bytes of every config the header names, from the `--config <path>` files the
/// caller staged (any number, matched by content). `repo apply` has no store to read
/// them from, and `config_digest` is derived from the construction document those bytes
/// carry — a header naming a config the caller did not stage cannot be applied.
fn staged_configs(
    header: &Header,
    flags: &Flags,
) -> tensorfs_core::err::Result<Vec<(String, Vec<u8>)>> {
    let mut staged: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (key, path) in flags {
        if key != "config" {
            continue;
        }
        let bytes = read(Path::new(path))?;
        staged.insert(tensorfs_core::sha256::hex_digest(&bytes), bytes);
    }
    header
        .configs
        .iter()
        .map(|(name, config)| match staged.get(&config.sha256) {
            Some(bytes) => Ok((name.clone(), bytes.clone())),
            None => Err(Refusal {
                code: Code::MISSING_FIELD,
                detail: format!(
                    "the header names config {name:?} (sha256:{}) but it was not staged: \
                     pass --config <path> for every config the header names",
                    config.sha256
                ),
            }),
        })
        .collect()
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
            let (header_bytes, configs, derived) = match (parsed.header(), flag(flags, "header")) {
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
                    let configs = match staged_configs(&header, flags) {
                        Ok(value) => value,
                        Err(error) => return bail(error),
                    };
                    let digests = match header.digests(&configs) {
                        Ok(value) => value,
                        Err(error) => return bail(error),
                    };
                    println!("topology_digest {}", digests.topology_digest);
                    println!("shape_digest    {}", digests.shape_digest);
                    if let Some(config) = &digests.config_digest {
                        println!("config_digest   {config}");
                    }
                    (Some(bytes), configs, Some(digests))
                }
                (None, None) => (None, Vec::new(), None),
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
            Some((manifest_bytes, header_bytes, configs, inventory, derived))
        }
        Mutation::RemoveCheckpoint { .. }
        | Mutation::UpdateRelease { .. }
        | Mutation::YankRelease { .. }
        | Mutation::DeleteRepository { .. } => None,
    };

    let derived = inputs.as_ref().and_then(|inputs| inputs.4.as_ref());
    let replacement = match repository::apply(current_bytes.as_deref(), &mutation, derived) {
        Ok(value) => value,
        Err(error) => return bail(error),
    };
    let (bytes, lines) = match replacement {
        Some(repository) => {
            let bytes = repository.canonical_bytes();
            let lines = match (&mutation, inputs) {
                (
                    Mutation::PutCheckpoint { manifest, .. },
                    Some((manifest_bytes, header, configs, inventory, _)),
                ) => {
                    let facts = match Census::validate_checkpoint_inputs(
                        manifest,
                        StagedCheckpoint {
                            manifest: &manifest_bytes,
                            header: header.as_deref(),
                            configs: &configs,
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
                    Some((manifest_bytes, header, configs, inventory, _)),
                ) => {
                    let facts = match Census::validate_checkpoint_inputs(
                        manifest,
                        StagedCheckpoint {
                            manifest: &manifest_bytes,
                            header: header.as_deref(),
                            configs: &configs,
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
    // A config is also a Manifest file entry; it is listed as `config` so a caller that
    // stages the closure for `repo apply` can tell which objects to pass as `--config`.
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
    for (_, entry) in manifest.entries() {
        if let Entry::File(object) = entry {
            insert(&mut refs, "file", object)?;
        }
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
