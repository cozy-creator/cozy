//! `cozy package lock|publish|yank`: authoring, on this computer, with uv. Package code never runs
//! here: the interface comes from Runtime's static describer.
//!
//! The project names no account: `{ index = "tensorhub" }` sources resolve to the account's index
//! on the command's Hub, written into a private copy. The authored files never change; `lock`
//! writes back only `uv.lock`.
use std::{
    collections::BTreeMap,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Duration,
};

use cozy_proto::{
    config::Hub,
    error::{Error, Kind},
};
use futures_util::{StreamExt as _, TryStreamExt as _};
use hub_client::types;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::{PackageRef, carried};
use crate::{Context, context::hub_error, output};

/// The uv index name a project's sources use for its own account's packages.
pub(super) const INDEX: &str = "tensorhub";

pub(super) fn refuse(code: &str, detail: impl Display) -> Error {
    Error::new(Kind::Validation, code, detail.to_string())
}

/// uv in `dir`, quiet and colorless.
pub(super) fn uv(dir: &Path) -> Command {
    let mut command = Command::new("uv");
    command.current_dir(dir).env("NO_COLOR", "1");
    command
}

/// Runs `command`; a failure is refused as `code` with the tool's own last words.
pub(super) fn run(command: &mut Command, code: &str) -> Result<String, Error> {
    let out = command.output().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => Error::new(
            Kind::Unavailable,
            "uv_not_found",
            "authoring packages needs uv, which is not on PATH",
        )
        .remedy("install it: https://docs.astral.sh/uv/getting-started/installation/"),
        _ => refuse(code, e),
    })?;
    if !out.status.success() {
        let said = [&out.stderr[..], &out.stdout[..]].concat();
        let said = String::from_utf8_lossy(&said);
        let tail: Vec<&str> = said.lines().filter(|l| !l.trim().is_empty()).collect();
        return Err(refuse(
            code,
            tail[tail.len().saturating_sub(12)..].join("\n"),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A package project, as its pyproject.toml and package.toml say.
pub(super) struct Project {
    pub root: PathBuf,
    pub name: String,
    pub version: String,
    pub requires_python: String,
    pub account_index: bool,
}

impl Project {
    /// The project in the current directory.
    fn here() -> Result<Self, Error> {
        Self::read(std::env::current_dir().map_err(|e| refuse("project_unreadable", e))?)
    }

    pub(super) fn read(root: PathBuf) -> Result<Self, Error> {
        let table = |file: &str| -> Result<toml::Table, Error> {
            let text = fs::read_to_string(root.join(file)).map_err(|e| {
                Error::new(
                    Kind::NotFound,
                    "project_file_missing",
                    format!("{file}: {e}"),
                )
                .remedy("run this in a package's directory (`cozy-runtime new <name>` makes one)")
            })?;
            toml::from_str(&text)
                .map_err(|e| refuse("project_file_invalid", format!("{file}: {e}")))
        };
        let pyproject = table("pyproject.toml")?;
        let package = table("package.toml")?;
        let field = |path: &[&str]| -> Option<&toml::Value> {
            let (first, rest) = path.split_first()?;
            rest.iter()
                .try_fold(pyproject.get(*first)?, |at, key| at.get(*key))
        };
        let text = |path: &[&str]| field(path).and_then(toml::Value::as_str).map(str::to_owned);
        let (Some(name), Some(version)) =
            (text(&["project", "name"]), text(&["project", "version"]))
        else {
            return Err(refuse(
                "project_identity_missing",
                "pyproject.toml's [project] needs a literal name and version",
            ));
        };
        let object = package
            .get("application")
            .and_then(|a| a.get("object"))
            .and_then(|o| o.as_str());
        let entry = field(&["project", "entry-points", "cozy.application"])
            .and_then(toml::Value::as_table)
            .map(|points| {
                points
                    .values()
                    .filter_map(toml::Value::as_str)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if object.is_none() || entry != [object.unwrap_or_default()] {
            return Err(refuse(
                "application_mismatch",
                format!(
                    "package.toml's [application] object ({}) must be pyproject's one cozy.application entry point ({})",
                    object.unwrap_or("absent"),
                    entry.join(", ")
                ),
            ));
        }
        let indexes = field(&["tool", "uv", "index"]).and_then(toml::Value::as_array);
        if indexes
            .into_iter()
            .flatten()
            .any(|i| i.get("name").and_then(|n| n.as_str()) == Some(INDEX))
        {
            return Err(refuse(
                "account_index_declared",
                "pyproject.toml declares the `tensorhub` index; cozy writes it for the Hub and account",
            )
            .remedy("delete that [[tool.uv.index]] table and keep `{ index = \"tensorhub\" }` in [tool.uv.sources]"));
        }
        let sources = field(&["tool", "uv", "sources"]).and_then(toml::Value::as_table);
        let names_index =
            |source: &toml::Value| source.get("index").and_then(|i| i.as_str()) == Some(INDEX);
        let account_index =
            sources
                .into_iter()
                .flat_map(|s| s.values())
                .any(|source| match source {
                    toml::Value::Array(alternatives) => alternatives.iter().any(names_index),
                    source => names_index(source),
                });
        Ok(Self {
            root,
            name: normalize(&name),
            version,
            requires_python: text(&["project", "requires-python"]).unwrap_or_default(),
            account_index,
        })
    }

    /// The Python a release runs on: `.python-version`, else requires-python's floor; its exact
    /// version as uv finds it (installing a managed one when none is here).
    fn python(&self) -> Result<Python, Error> {
        let pinned = fs::read_to_string(self.root.join(".python-version")).ok();
        let floor = self.requires_python.split(',').find_map(|clause| {
            let version = clause
                .trim()
                .strip_prefix(">=")
                .or_else(|| clause.trim().strip_prefix("~="))?;
            Some(
                version
                    .trim()
                    .splitn(3, '.')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join("."),
            )
        });
        let request = pinned
            .as_deref()
            .and_then(|text| text.lines().next())
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .or(floor)
            .unwrap_or_else(|| "3.12".into());
        let find = || {
            run(
                uv(&self.root).args(["python", "find", "--show-version", request.as_str()]),
                "python_not_found",
            )
        };
        let full = match find() {
            Ok(version) => version,
            Err(_) => {
                run(
                    uv(&self.root).args(["python", "install", request.as_str()]),
                    "python_install_failed",
                )?;
                find()?
            }
        };
        Ok(Python {
            request,
            full: full.trim().to_owned(),
        })
    }
}

struct Python {
    /// What uv is asked for (`3.12`).
    request: String,
    /// The exact version a release records (`3.12.12`).
    full: String,
}

/// PEP 503: lowercase, runs of `-_.` become one `-`.
fn normalize(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars() {
        match c {
            '-' | '_' | '.' if out.ends_with('-') => {}
            '-' | '_' | '.' => out.push('-'),
            c => out.push(c.to_ascii_lowercase()),
        }
    }
    out
}

/// A private copy of the project to lock and build in.
pub(super) struct Stage {
    _dir: tempfile::TempDir,
    pub root: PathBuf,
    pub src: PathBuf,
}

impl Stage {
    fn new(project: &Project) -> Result<Self, Error> {
        let dir = tempfile::tempdir().map_err(|e| refuse("stage_unavailable", e))?;
        let root = dir.path().to_path_buf();
        let src = root.join("src");
        for file in source_files(&project.root)? {
            let to = src.join(&file);
            fs::create_dir_all(to.parent().unwrap_or(&src))
                .and_then(|()| fs::copy(project.root.join(&file), &to))
                .map_err(|e| refuse("stage_unavailable", e))?;
        }
        Ok(Self {
            _dir: dir,
            root,
            src,
        })
    }

    /// Points the copy's `tensorhub` index at `url`.
    fn bind(&self, url: &str) -> Result<(), Error> {
        let path = self.src.join("pyproject.toml");
        let mut text = fs::read_to_string(&path).map_err(|e| refuse("stage_unavailable", e))?;
        text.push_str(&format!(
            "\n[[tool.uv.index]]\nname = \"{INDEX}\"\nurl = \"{url}\"\nexplicit = true\n"
        ));
        fs::write(&path, text).map_err(|e| refuse("stage_unavailable", e))
    }

    /// Relocks the copy against its bound index with `pins` as constraints, then without them:
    /// a lock made for another index moves to this one keeping each exact version (or fails
    /// naming the pin), and the copy publishes with no constraint it did not author.
    fn relock_pinned(&self, python: &str, pins: &BTreeMap<String, String>) -> Result<(), Error> {
        let path = self.src.join("pyproject.toml");
        let authored = fs::read_to_string(&path).map_err(|e| refuse("stage_unavailable", e))?;
        let mut doc: toml_edit::DocumentMut =
            authored.parse().map_err(|e| refuse("project_invalid", e))?;
        let constraints = doc["tool"]["uv"]["constraint-dependencies"]
            .or_insert(toml_edit::value(toml_edit::Array::new()))
            .as_array_mut()
            .ok_or_else(|| {
                refuse(
                    "project_invalid",
                    "tool.uv.constraint-dependencies is not a list",
                )
            })?;
        for (name, version) in pins {
            constraints.push(format!("{name}=={version}"));
        }
        fs::write(&path, doc.to_string()).map_err(|e| refuse("stage_unavailable", e))?;
        let locked = run(
            uv(&self.src).args(["lock", "--no-progress", "--python", python]),
            "lock_failed",
        );
        fs::write(&path, authored).map_err(|e| refuse("stage_unavailable", e))?;
        locked.map(drop)
    }
}

/// The project's source files, relative to `root`: what is never source is skipped, and what must
/// never ship (credentials, keys, bytecode, model weights) is refused.
pub(super) fn source_files(root: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut files = vec![];
    walk(root, Path::new(""), &mut files)?;
    files.sort();
    Ok(files)
}

fn walk(root: &Path, rel: &Path, files: &mut Vec<PathBuf>) -> Result<(), Error> {
    const SKIP: &str = ".git .hg .svn .jj __pycache__ .mypy_cache .ruff_cache .pytest_cache .tox";
    const SKIP_AT_ROOT: &str = ".venv venv node_modules .idea .vscode build dist";
    const SECRET_DIRS: &str = ".aws .ssh credentials secrets";
    const SECRET_FILES: &str =
        ".netrc .npmrc .pypirc credentials.json service-account.json id_ed25519";
    const REFUSED: &str =
        ".pyc .pyo .pem .key .p12 .pfx .jks .pickle .pkl .pt .ckpt .safetensors .onnx .gguf";
    let listed = |list: &str, name: &str| list.split(' ').any(|item| item == name);
    let entries = fs::read_dir(root.join(rel)).map_err(|e| refuse("project_unreadable", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| refuse("project_unreadable", e))?;
        let name = entry.file_name().to_string_lossy().to_lowercase();
        let path = rel.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|e| refuse("project_unreadable", e))?;
        let refused = |why: &str| {
            refuse(
                "package_source_refused",
                format!("{} {why}", path.display()),
            )
            .remedy(
                "move credentials, keys, bytecode and model weights out of the package directory",
            )
        };
        if kind.is_symlink() {
            return Err(refused("is a symlink"));
        }
        if kind.is_dir() {
            if listed(SECRET_DIRS, &name) {
                return Err(refused("holds credentials"));
            }
            let at_root = rel.as_os_str().is_empty() && listed(SKIP_AT_ROOT, &name);
            if !listed(SKIP, &name) && !name.ends_with(".egg-info") && !at_root {
                walk(root, &path, files)?;
            }
            continue;
        }
        let secret = name == ".env"
            || name.starts_with(".env.")
            || name.starts_with("id_rsa")
            || listed(SECRET_FILES, &name);
        if secret || REFUSED.split(' ').any(|suffix| name.ends_with(suffix)) {
            return Err(refused("must never ship in a package"));
        }
        let scratch = name.ends_with(".swp")
            || name.ends_with('~')
            || name.starts_with(".#")
            || name == ".ds_store";
        if !scratch && name != "package-interface.json" {
            files.push(path);
        }
    }
    Ok(())
}

/// Each locked package's version.
/// The packages a lock takes from a Tensorhub account index, by exact version.
fn index_pins(lock: &str) -> Result<BTreeMap<String, String>, Error> {
    #[derive(Deserialize)]
    struct Lock {
        #[serde(default)]
        package: Vec<Locked>,
    }
    #[derive(Deserialize)]
    struct Locked {
        name: String,
        #[serde(default)]
        version: String,
        #[serde(default)]
        source: toml::Table,
    }
    let lock: Lock = toml::from_str(lock).map_err(|e| refuse("lock_invalid", e))?;
    Ok(lock
        .package
        .into_iter()
        .filter(|p| {
            p.source
                .get("registry")
                .and_then(|r| r.as_str())
                .is_some_and(|r| r.contains("/v1/index/"))
        })
        .map(|p| (p.name, p.version))
        .collect())
}

fn versions(lock: &str) -> Result<BTreeMap<String, String>, Error> {
    #[derive(Deserialize)]
    struct Lock {
        #[serde(default)]
        package: Vec<Locked>,
    }
    #[derive(Deserialize)]
    struct Locked {
        name: String,
        #[serde(default)]
        version: String,
        #[serde(default)]
        source: toml::Table,
    }
    let lock: Lock = toml::from_str(lock).map_err(|e| refuse("lock_invalid", e))?;
    let mut out = BTreeMap::new();
    for package in lock.package {
        let local = ["directory", "path", "virtual", "editable"]
            .iter()
            .find(|k| package.source.contains_key(**k));
        if let Some(kind) = local
            && package.source.get(*kind).and_then(|v| v.as_str()) != Some(".")
        {
            return Err(refuse(
                "package_local_dependency",
                format!("{} is a local {kind} dependency, which a release cannot carry", package.name),
            )
            .remedy(format!(
                "publish {} to your account and name it with `{{ index = \"{INDEX}\" }}` in [tool.uv.sources]",
                package.name
            )));
        }
        out.insert(package.name, package.version);
    }
    Ok(out)
}

/// The account's index on `hub` at the origin the Hub advertises (RFC 9728 `resource`): the one
/// its publish check holds a uv.lock to, wherever this computer reaches the Hub. A Hub that
/// advertises none takes no publication, and its configured origin stands.
pub(super) async fn index_url(
    client: &hub_client::Client,
    hub: &Hub,
    account: &str,
) -> Result<String, Error> {
    let origin = match client.get_protected_resource_metadata().await {
        Ok(metadata) => metadata.into_inner().resource,
        Err(e) => match hub_error(e) {
            e if e.code == "capability.not_served" => hub.origin.clone(),
            e => return Err(e),
        },
    };
    Ok(format!(
        "{}/v1/index/{account}/simple/",
        origin.trim_end_matches('/')
    ))
}

pub(super) async fn account(client: &hub_client::Client) -> Result<String, Error> {
    Ok(client
        .get_current_account()
        .await
        .map_err(hub_error)?
        .into_inner()
        .name)
}

#[derive(Serialize)]
struct Locked {
    lock: PathBuf,
    account_index: Option<String>,
    changed: bool,
}

pub async fn lock(ctx: &Context, upgrade: &[String]) -> Result<ExitCode, Error> {
    let project = Project::here()?;
    let stage = Stage::new(&project)?;
    let account_index = match project.account_index {
        false => None,
        true => {
            let hub = ctx.hub()?;
            let client = ctx.hub_signed_in(&hub)?;
            let account = account(&client).await?;
            let url = index_url(&client, &hub, &account).await?;
            stage.bind(&url)?;
            Some(url)
        }
    };
    let python = project.python()?;
    let mut command = uv(&stage.src);
    command.args(["lock", "--no-progress", "--python", python.request.as_str()]);
    for package in upgrade {
        command.args(["--upgrade-package", package.as_str()]);
    }
    run(&mut command, "lock_failed")?;
    let locked = fs::read(stage.src.join("uv.lock")).map_err(|e| refuse("lock_failed", e))?;
    versions(&String::from_utf8_lossy(&locked))?;
    let path = project.root.join("uv.lock");
    let changed = fs::read(&path).ok().as_deref() != Some(&locked[..]);
    if changed {
        let staged = project.root.join(".uv.lock.cozy");
        fs::write(&staged, &locked)
            .and_then(|()| fs::rename(&staged, &path))
            .map_err(|e| refuse("lock_unwritable", e))?;
    }
    let result = Locked {
        lock: path,
        account_index,
        changed,
    };
    match ctx.json {
        true => output::json(&result),
        false => output::Record::new()
            .field("lock", result.lock.display())
            .field(
                "account index",
                result.account_index.as_deref().unwrap_or("none"),
            )
            .field("changed", result.changed)
            .print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Serialize)]
struct Published {
    package: String,
    release: String,
    status: &'static str,
    changed: bool,
    uploaded_bytes: u64,
    hub: String,
    warnings: Vec<String>,
}

impl Published {
    fn print(&self, ctx: &Context) {
        if ctx.json {
            return output::json(self);
        }
        let mut record = output::Record::new()
            .field("package", format!("{}@{}", self.package, self.release))
            .field("status", self.status)
            .field("uploaded", output::bytes(self.uploaded_bytes))
            .field("hub", &self.hub);
        for warning in &self.warnings {
            record = record.note(format!("warning: {warning}"));
        }
        record.print(ctx);
    }
}

/// One file of a release: where it goes and where it is here.
struct File {
    path: String,
    local: PathBuf,
    digest: String,
    length: i64,
}

impl File {
    fn new(path: String, local: PathBuf) -> Result<Self, Error> {
        let bytes = fs::read(&local).map_err(|e| refuse("release_file_unreadable", e))?;
        Ok(Self {
            path,
            local,
            digest: format!("sha256:{}", hex::encode(Sha256::digest(&bytes))),
            length: bytes.len() as i64,
        })
    }
}

pub async fn publish(ctx: &Context) -> Result<ExitCode, Error> {
    let project = Project::here()?;
    let hub = ctx.hub()?;
    let client = ctx.hub_signed_in(&hub)?;
    let account = account(&client).await?;
    let mut result = Published {
        package: format!("{account}/{}", project.name),
        release: project.version.clone(),
        status: "already published",
        changed: false,
        uploaded_bytes: 0,
        hub: hub.label().to_owned(),
        warnings: vec![],
    };
    match client
        .get_package_release(&account, &project.name, &project.version)
        .await
    {
        Ok(_) => {
            result.print(ctx);
            output::next(
                ctx,
                &["bump [project] version in pyproject.toml to publish a change"],
            );
            return Ok(ExitCode::SUCCESS);
        }
        Err(e) => match hub_error(e) {
            e if e.kind == Kind::NotFound => {}
            e => return Err(e),
        },
    }

    let stage = Stage::new(&project)?;
    let committed = fs::read_to_string(project.root.join("uv.lock")).map_err(|_| {
        Error::new(
            Kind::Validation,
            "package_unlocked",
            "this project has no uv.lock",
        )
        .next("cozy package lock")
    })?;
    // Locking the copy is a no-op for a lock made for this Hub and account. For another, it moves
    // the account's packages to this index pinned to the versions the lock holds: a pin this Hub
    // cannot meet is refused by name, never re-resolved to a newer release.
    let python = project.python()?;
    if project.account_index {
        let url = index_url(&client, &hub, &account).await?;
        stage.bind(&url)?;
        let pins = index_pins(&committed)?;
        if !pins.is_empty() && !committed.contains(&url) {
            stage
                .relock_pinned(python.request.as_str(), &pins)
                .map_err(|e| unmet(e, &pins, &account, &hub))?;
        }
    }
    run(
        uv(&stage.src).args(["lock", "--no-progress", "--python", python.request.as_str()]),
        "lock_failed",
    )?;
    let bound =
        fs::read_to_string(stage.src.join("uv.lock")).map_err(|e| refuse("lock_failed", e))?;
    let (before, after) = (versions(&committed)?, versions(&bound)?);
    if before != after {
        let detail = before
            .iter()
            .find(|(name, version)| after.get(*name) != Some(*version))
            .map(|(name, version)| match after.get(name) {
                Some(now) => format!("uv re-resolved {name} from {version} to {now}"),
                None => format!("uv dropped {name} {version}"),
            })
            .unwrap_or_else(|| "uv added a package the lock does not hold".into());
        return Err(refuse(
            "account_index_lock_drift",
            format!(
                "uv.lock does not hold for {} on {}: {detail}",
                result.package,
                hub.label()
            ),
        )
        .remedy("publish the locked versions of your own dependencies to this Hub first, or relock")
        .next(format!("cozy package lock --tensorhub {}", hub.label())));
    }

    let dist = stage.root.join("dist");
    run(
        uv(&stage.src)
            .args([
                "build",
                "--no-progress",
                "--python",
                python.request.as_str(),
                "--out-dir",
            ])
            .arg(&dist),
        "build_failed",
    )?;
    let git_wheels = carried::git_wheels(&stage, &python.request)?;
    let interface = describe(&stage, &python, after.get("cozy-runtime"), &account)?;
    let interface_path = stage.root.join("package-interface.json");
    fs::write(&interface_path, &interface).map_err(|e| refuse("stage_unavailable", e))?;

    let wheel = carried::single(&dist, ".whl")?;
    let sdist = carried::single(&dist, ".tar.gz")?;
    let named = |dir: &str, local: &Path| {
        format!(
            "{dir}/{}",
            local.file_name().unwrap_or_default().to_string_lossy()
        )
    };
    let mut files = vec![
        File::new(named("artifacts/project", &wheel), wheel.clone())?,
        File::new(named("artifacts/source", &sdist), sdist.clone())?,
        File::new("metadata/package-interface.json".into(), interface_path)?,
    ];
    for document in ["package.toml", "pyproject.toml", "uv.lock"] {
        files.push(File::new(document.into(), stage.src.join(document))?);
    }
    for local in git_wheels {
        files.push(File::new(named("artifacts/dependencies", &local), local)?);
    }

    let declaration = types::PackageReleaseDeclaration {
        files: files
            .iter()
            .map(|f| types::PackageDeclaredFile {
                digest: f.digest.clone(),
                length: f.length,
                path: f.path.clone(),
            })
            .collect(),
    };
    let (org, name, release) = (&account, &project.name, &project.version);
    let grants = client
        .begin_package_release(org, name, release, &declaration)
        .await
        .map_err(hub_error)?
        .into_inner();
    result.uploaded_bytes = upload(ctx.http(), &files, &grants.files).await?;
    let finalize = types::PackageReleaseFinalize {
        publication_id: grants.publication_id.clone(),
        python_version: Some(python.full.clone()),
    };
    let mut state = client
        .finalize_package_release(org, name, release, &finalize)
        .await
        .map_err(hub_error)?
        .into_inner();
    // Finalization re-hashes every file on the Hub; its state ends the wait.
    while state.state != "committed" && state.state != "failed" {
        tokio::time::sleep(Duration::from_secs(2)).await;
        state = client
            .get_package_finalization(org, name, release)
            .await
            .map_err(hub_error)?
            .into_inner();
    }
    if state.state == "failed" {
        let said = |key: &str| {
            state
                .error
                .as_ref()
                .and_then(|e| e.get(key))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
        };
        let mut error = refuse(
            "package_release_failed",
            said("message").unwrap_or_else(|| "the Hub refused the release".into()),
        );
        if let Some(code) = said("code") {
            error = error.cause(code);
        }
        if let Some(remedy) = said("remedy") {
            error = error.remedy(remedy);
        }
        return Err(error);
    }
    result.status = "published";
    result.changed = true;
    result.warnings = state.warnings;
    result.print(ctx);
    Ok(ExitCode::SUCCESS)
}

/// A relock that could not keep the lock's account-index pins on `hub`: the pin uv names, by
/// package, version and Hub, when it names one; otherwise uv's own words.
fn unmet(error: Error, pins: &BTreeMap<String, String>, account: &str, hub: &Hub) -> Error {
    let named = pins
        .iter()
        .find(|(name, version)| error.message.contains(&format!("{name}=={version}")));
    match named {
        Some((name, version)) => refuse(
            "account_index_pin_unpublished",
            format!(
                "uv.lock pins {account}/{name} {version}, which {} does not publish",
                hub.label()
            ),
        )
        .remedy(format!(
            "publish {name} {version} to this Hub as {account} first, or relock against it"
        ))
        .next(format!("cozy package lock --tensorhub {}", hub.label())),
        None => error,
    }
}

/// The interface Runtime's static describer reads from the staged source, with the publishing
/// account written into org-relative model defaults; refused when installs could not read it.
fn describe(
    stage: &Stage,
    python: &Python,
    runtime: Option<&String>,
    account: &str,
) -> Result<Vec<u8>, Error> {
    let from = match runtime {
        Some(version) => format!("cozy-runtime=={version}"),
        None => "cozy-runtime".into(),
    };
    let said = run(
        uv(&stage.src)
            .args([
                "tool",
                "run",
                "--no-progress",
                "--python",
                python.request.as_str(),
                "--from",
                from.as_str(),
            ])
            .args(["cozy-runtime", "--json", "--dir"])
            .arg(&stage.src)
            .arg("describe"),
        "describe_failed",
    )?;
    let mut interface: Value =
        serde_json::from_str(&said).map_err(|e| refuse("describe_failed", e))?;
    let mut qualified = false;
    // `get_mut`, not indexing: indexing a missing key would write a null into the document.
    let callables = interface
        .get_mut("entrypoints")
        .and_then(Value::as_array_mut);
    for callable in callables.into_iter().flatten() {
        let slots = callable.get_mut("models").and_then(Value::as_array_mut);
        for slot in slots.into_iter().flatten() {
            let Some(default) = slot.get("default").and_then(Value::as_str) else {
                continue;
            };
            if default
                .split_once('@')
                .is_some_and(|(model, _)| !model.contains('/'))
            {
                let named = format!("{account}/{default}");
                slot["default"] = named.into();
                qualified = true;
            }
        }
    }
    let readable = crate::run::interface::Interface::parse(&interface).map_err(|why| {
        refuse(
            "package_interface_invalid",
            format!("installs would refuse it: {why}"),
        )
    })?;
    if let Some((callable, why)) = readable.unavailable.first() {
        return Err(refuse(
            "package_interface_invalid",
            format!("installs could not call {callable}: {why}"),
        ));
    }
    Ok(match qualified {
        true => serde_json::to_vec(&interface).map_err(|e| refuse("describe_failed", e))?,
        false => said.trim_end().as_bytes().to_vec(),
    })
}

/// PUTs the files the Hub does not hold yet, 16 at a time; answers the bytes sent. A 412 means
/// the store already holds those exact bytes.
async fn upload(
    http: &reqwest::Client,
    files: &[File],
    grants: &[types::PackageFileGrant],
) -> Result<u64, Error> {
    let sends = grants.iter().filter(|g| !g.present).map(|grant| {
        let http = http.clone();
        let file = files.iter().find(|f| f.path == grant.path);
        async move {
            let (Some(file), Some(target)) = (file, &grant.upload) else {
                return Err(refuse(
                    "release_grant_invalid",
                    format!("the Hub granted {} without an upload", grant.path),
                ));
            };
            let bytes = tokio::fs::read(&file.local)
                .await
                .map_err(|e| refuse("release_file_unreadable", e))?;
            let length = bytes.len() as u64;
            let mut request = http.put(&target.url).body(bytes);
            for (name, value) in &target.required_headers {
                request = request.header(name, value);
            }
            let answer = request.send().await.map_err(|e| {
                Error::new(
                    Kind::Unavailable,
                    "release_upload_failed",
                    format!("{}: {e}", file.path),
                )
            })?;
            match answer.status().as_u16() {
                200..=299 => Ok(length),
                412 => Ok(0),
                status => Err(Error::new(
                    Kind::Unavailable,
                    "release_upload_failed",
                    format!("{}: the store answered HTTP {status}", file.path),
                )),
            }
        }
    });
    futures_util::stream::iter(sends)
        .buffer_unordered(16)
        .try_fold(0, |sum, sent| async move { Ok::<u64, Error>(sum + sent) })
        .await
}

#[derive(Serialize)]
struct Yanked {
    package: String,
    release: String,
    changed: bool,
    yanked_at: String,
}

pub async fn yank(ctx: &Context, package: &PackageRef) -> Result<ExitCode, Error> {
    let Some(release) = &package.release else {
        return Err(Error::new(
            Kind::Usage,
            "yank_release_required",
            "name the release to yank: org/name@release",
        ));
    };
    let hub = ctx.hub()?;
    let yanked = ctx
        .hub_signed_in(&hub)?
        .yank_package_release(
            &package.org,
            &package.name,
            release,
            &types::Empty(Default::default()),
        )
        .await
        .map_err(hub_error)?
        .into_inner();
    let result = Yanked {
        package: package.package(),
        release: yanked.release,
        changed: yanked.changed,
        yanked_at: super::stamp(&yanked.yanked_at),
    };
    match ctx.json {
        true => output::json(&result),
        false => output::Record::new()
            .field("package", format!("{}@{}", result.package, result.release))
            .field(
                "status",
                if result.changed {
                    "yanked"
                } else {
                    "already yanked"
                },
            )
            .field("yanked at", &result.yanked_at)
            .note("Machines that hold it keep it; nothing new installs it.")
            .print(ctx),
    }
    Ok(ExitCode::SUCCESS)
}
