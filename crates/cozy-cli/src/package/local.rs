//! Local code as a machine takes it: TensorD's `LocalSource`, a manifest naming each source file
//! by its content. A directory travels with the path dependencies its sources name, under their
//! common directory; a PEP 723 script becomes a generated project (scripts are packages). Nothing
//! is locked here: a project without a uv.lock is resolved on the machine. This computer's machine
//! reads the files in place, so only the manifest and the carried wheels travel; another receives
//! the files too, and Calcifer writes only the bytes it lacks.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use cozy_proto::{calcifer::StagedInput, error::Error, ids::Digest, machine as v1};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::author::{INDEX, Project, account, index_url, refuse, source_files};
use crate::Context;

#[derive(Serialize)]
struct Manifest {
    package: String,
    release: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    python_requires: String,
    files: Vec<Member>,
    #[serde(skip_serializing_if = "String::is_empty")]
    project: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    locals: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    path: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    callees: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    indexes: BTreeMap<String, String>,
    /// Wheels a path source names (`{ path = "dist/x.whl" }`), carried: no index the machine
    /// reaches serves them.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    wheels: Vec<Member>,
}

#[derive(Serialize)]
struct Member {
    name: String,
    digest: String,
    length: u64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    executable: bool,
}

/// What a run names to install local code: its source, the objects Calcifer writes first, and the
/// `local/<name>` package it is. Kept until the submit returns: it holds the manifest file.
pub(crate) struct Local {
    pub package: String,
    pub source: v1::LocalSource,
    pub inputs: Vec<StagedInput>,
    /// What the machine's environment is built from, apart from the project's own source: its
    /// metadata and lock, its path dependencies, carried wheels and indexes.
    pub environment: Digest,
    _manifest: tempfile::TempDir,
}

/// `path` (a project directory or a PEP 723 script) as a machine installs it. `in_place`: the
/// machine is this computer's, which reads the files where they are.
pub(crate) async fn source(ctx: &Context, path: &Path, in_place: bool) -> Result<Local, Error> {
    let path = fs::canonicalize(path)
        .map_err(|e| refuse("project_unreadable", format!("{}: {e}", path.display())))?;
    let project_dir = match path.is_file() {
        true => script(ctx, &path)?,
        false => path,
    };
    let project = Project::read(project_dir.clone())?;
    let locals = path_dependencies(&project_dir)?;
    let root = common(
        project_dir.as_path(),
        locals.iter().map(|l| {
            if l.is_file() {
                l.parent().unwrap_or(l)
            } else {
                l
            }
        }),
    );
    let relative = |p: &Path| {
        p.strip_prefix(&root)
            .unwrap_or(p)
            .to_string_lossy()
            .replace('\\', "/")
    };
    let mut names: Vec<PathBuf> = source_files(&project_dir)?
        .into_iter()
        .map(|f| project_dir.join(f))
        .collect();
    for local in &locals {
        match local.is_file() {
            true => names.push(local.clone()),
            false => names.extend(source_files(local)?.into_iter().map(|f| local.join(f))),
        }
    }
    // A path source inside the project (`deps/x.whl`) is one of its files already: listed twice,
    // its wheel was carried twice and the machine's second copy was refused.
    let mut seen = std::collections::HashSet::new();
    names.retain(|name| seen.insert(name.clone()));
    let carried = |file: &Path| {
        locals.iter().any(|l| l == file) && file.extension().is_some_and(|e| e == "whl")
    };
    let (mut files, mut wheels, mut inputs) = (vec![], vec![], vec![]);
    for file in names {
        let bytes = fs::read(&file).map_err(|e| refuse("project_unreadable", e))?;
        let digest = Digest::of(&bytes);
        let length = bytes.len() as u64;
        files.push(Member {
            name: relative(&file),
            digest: digest.to_string(),
            length,
            executable: executable(&file),
        });
        // A carried wheel is an object the machine holds, even one that reads the tree in place.
        if carried(&file) {
            let name = file
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            wheels.push(Member {
                name,
                digest: digest.to_string(),
                length,
                executable: false,
            });
        }
        if !in_place || carried(&file) {
            inputs.push(StagedInput {
                path: file,
                digest,
                length,
            });
        }
    }
    let mut indexes = BTreeMap::new();
    if project.account_index {
        let hub = ctx.hub()?;
        let client = ctx.hub_signed_in(&hub)?;
        let account = account(&client).await?;
        indexes.insert(INDEX.to_owned(), index_url(&client, &hub, &account).await?);
    }
    let manifest = Manifest {
        package: format!("local/{}", project.name),
        release: project.version.clone(),
        python_requires: project.requires_python.clone(),
        files,
        project: relative(&project_dir),
        locals: locals.iter().map(|l| relative(l)).collect(),
        path: if in_place {
            root.to_string_lossy().into_owned()
        } else {
            String::new()
        },
        callees: callees(&project_dir)?,
        indexes,
        wheels,
    };
    let own = match manifest.project.is_empty() {
        true => String::new(),
        false => format!("{}/", manifest.project),
    };
    let builds = |m: &&Member| {
        let beside = !m.name.starts_with(&own)
            || manifest
                .locals
                .iter()
                .any(|l| m.name == *l || m.name.starts_with(&format!("{l}/")));
        beside || m.name == format!("{own}pyproject.toml") || m.name == format!("{own}uv.lock")
    };
    let built: Vec<(&str, &str)> = (manifest.files.iter().filter(builds))
        .chain(&manifest.wheels)
        .map(|m| (m.name.as_str(), m.digest.as_str()))
        .collect();
    let environment = serde_json::json!({"files": built, "indexes": manifest.indexes,
        "python": manifest.python_requires});
    let environment = Digest::of(environment.to_string().as_bytes());
    let document = serde_json::to_vec(&manifest).map_err(|e| refuse("local_source_invalid", e))?;
    let held = tempfile::tempdir().map_err(|e| refuse("stage_unavailable", e))?;
    let written = held.path().join("manifest.json");
    fs::write(&written, &document).map_err(|e| refuse("stage_unavailable", e))?;
    let digest = Digest::of(&document);
    inputs.push(StagedInput {
        path: written,
        digest,
        length: document.len() as u64,
    });
    Ok(Local {
        package: manifest.package,
        source: v1::LocalSource {
            manifest: digest.to_string(),
        },
        inputs,
        environment,
        _manifest: held,
    })
}

/// The local path dependencies the project's sources name (`{ path = "../lib" }`), each a
/// directory or a file.
fn path_dependencies(project: &Path) -> Result<Vec<PathBuf>, Error> {
    let text = fs::read_to_string(project.join("pyproject.toml"))
        .map_err(|e| refuse("project_unreadable", e))?;
    let pyproject: toml::Table =
        toml::from_str(&text).map_err(|e| refuse("project_file_invalid", e))?;
    let sources = pyproject
        .get("tool")
        .and_then(|t| t.get("uv"))
        .and_then(|u| u.get("sources"))
        .and_then(|s| s.as_table());
    let mut out = vec![];
    for source in sources.into_iter().flat_map(|s| s.values()) {
        let Some(path) = source.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        let dependency = fs::canonicalize(project.join(path))
            .map_err(|e| refuse("local_dependency_unreadable", format!("{path}: {e}")))?;
        out.push(dependency);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// The deepest directory holding `first` and every one of `rest`.
fn common<'a>(first: &'a Path, rest: impl Iterator<Item = &'a Path>) -> PathBuf {
    let mut root = first.to_path_buf();
    for dir in rest {
        while !dir.starts_with(&root) {
            match root.parent() {
                Some(parent) => root = parent.to_path_buf(),
                None => break,
            }
        }
    }
    root
}

/// The distributions the lock takes from an account index, as the packages they are
/// (`qwen-image-2` → `fidika/qwen-image-2`): their callables are that package's.
fn callees(project: &Path) -> Result<BTreeMap<String, String>, Error> {
    #[derive(Deserialize)]
    struct Lock {
        #[serde(default)]
        package: Vec<Locked>,
    }
    #[derive(Deserialize)]
    struct Locked {
        name: String,
        #[serde(default)]
        source: toml::Table,
    }
    let Ok(text) = fs::read_to_string(project.join("uv.lock")) else {
        return Ok(BTreeMap::new());
    };
    let lock: Lock = toml::from_str(&text).map_err(|e| refuse("lock_invalid", e))?;
    let callee = |p: &Locked| {
        let registry = p.source.get("registry")?.as_str()?;
        let org = registry
            .split_once("/v1/index/")?
            .1
            .strip_suffix("/simple/")?;
        Some((p.name.clone(), format!("{org}/{}", p.name)))
    };
    Ok(lock.package.iter().filter_map(callee).collect())
}

/// A PEP 723 script as the project it is: generated under `~/.cozy/scripts`, named by the
/// script's content, so an unchanged script is the same project.
fn script(ctx: &Context, path: &Path) -> Result<PathBuf, Error> {
    let raw = fs::read(path).map_err(|e| refuse("script_unreadable", e))?;
    let mut metadata = pep723(&String::from_utf8_lossy(&raw))?;
    let name = format!("cozy-script-{}", &hex::encode(Sha256::digest(&raw))[..24]);
    let root = ctx.home.root().join("scripts").join(&name);
    let mut dependencies = metadata
        .remove("dependencies")
        .and_then(|d| d.as_array().cloned())
        .unwrap_or_default();
    let names_runtime = dependencies.iter().filter_map(|d| d.as_str()).any(|d| {
        d.trim_start()
            .to_ascii_lowercase()
            .replace('_', "-")
            .starts_with("cozy-runtime")
    });
    if !names_runtime {
        dependencies.push("cozy-runtime>=0.27".into());
    }
    let requires = metadata
        .remove("requires-python")
        .and_then(|r| r.as_str().map(str::to_owned));
    let mut tool = metadata
        .remove("tool")
        .and_then(|t| t.as_table().cloned())
        .unwrap_or_default();
    // A path source names a directory beside the script: relative to the generated project
    // instead, so the project and its libraries travel to a machine as one tree.
    let sources = tool
        .get_mut("uv")
        .and_then(|u| u.get_mut("sources"))
        .and_then(|s| s.as_table_mut());
    for (_, source) in sources.into_iter().flat_map(|s| s.iter_mut()) {
        if let Some(beside) = source.get("path").and_then(|p| p.as_str()) {
            let absolute = fs::canonicalize(path.parent().unwrap_or(path).join(beside))
                .map_err(|e| refuse("local_dependency_unreadable", format!("{beside}: {e}")))?;
            source["path"] = from(&root, &absolute).into();
        }
    }
    // Hatchling builds the two modules, and admits git or URL dependencies as uv locks them.
    let hatch: toml::Table = toml::from_str(
        "metadata.allow-direct-references = true\n\
         build.targets.wheel.only-include = [\"cozy_script.py\", \"cozy_script_entry.py\"]\n",
    )
    .expect("a constant table");
    tool.insert("hatch".into(), hatch.into());
    let pyproject = Pyproject {
        project: ScriptProject {
            name: name.clone(),
            version: "0.0.0",
            requires_python: requires.unwrap_or_else(|| ">=3.12".into()),
            dependencies,
            entry_points: BTreeMap::from([(
                "cozy.application",
                BTreeMap::from([("default", "cozy_script_entry:app")]),
            )]),
        },
        build_system: BuildSystem {
            requires: vec!["hatchling>=1.25"],
            build_backend: "hatchling.build",
        },
        tool,
    };
    let files = [
        (
            "pyproject.toml",
            toml::to_string(&pyproject).map_err(|e| refuse("script_metadata_invalid", e))?,
        ),
        (
            "package.toml",
            "[application]\nobject = \"cozy_script_entry:app\"\n".into(),
        ),
        (
            "cozy_script_entry.py",
            "from cozy_runtime.author import script_app\napp = script_app(\"cozy_script\")\n"
                .into(),
        ),
    ];
    fs::create_dir_all(&root).map_err(|e| refuse("script_unwritable", e))?;
    fs::write(root.join("cozy_script.py"), &raw).map_err(|e| refuse("script_unwritable", e))?;
    for (file, text) in files {
        fs::write(root.join(file), text).map_err(|e| refuse("script_unwritable", e))?;
    }
    Ok(root)
}

#[derive(Serialize)]
struct Pyproject {
    project: ScriptProject,
    #[serde(rename = "build-system")]
    build_system: BuildSystem,
    tool: toml::Table,
}

#[derive(Serialize)]
struct ScriptProject {
    name: String,
    version: &'static str,
    #[serde(rename = "requires-python")]
    requires_python: String,
    dependencies: Vec<toml::Value>,
    #[serde(rename = "entry-points")]
    entry_points: BTreeMap<&'static str, BTreeMap<&'static str, &'static str>>,
}

#[derive(Serialize)]
struct BuildSystem {
    requires: Vec<&'static str>,
    #[serde(rename = "build-backend")]
    build_backend: &'static str,
}

/// `to` as a slash path relative to the directory `dir`.
fn from(dir: &Path, to: &Path) -> String {
    let shared = dir
        .components()
        .zip(to.components())
        .take_while(|(a, b)| a == b)
        .count();
    let up = dir.components().count() - shared;
    let down = to
        .components()
        .skip(shared)
        .map(|c| c.as_os_str().to_string_lossy().into_owned());
    std::iter::repeat_n("..".to_owned(), up)
        .chain(down)
        .collect::<Vec<_>>()
        .join("/")
}

/// The script's `# /// script` metadata block (PEP 723) as TOML; none is an empty table.
fn pep723(text: &str) -> Result<toml::Table, Error> {
    let mut lines = text
        .lines()
        .skip_while(|line| line.trim_end() != "# /// script");
    if lines.next().is_none() {
        return Ok(toml::Table::new());
    }
    let mut block = String::new();
    for line in lines {
        match line.trim_end() {
            "# ///" => {
                return toml::from_str(&block).map_err(|e| refuse("script_metadata_invalid", e));
            }
            "#" => block.push('\n'),
            line => match line.strip_prefix("# ") {
                Some(content) => block.push_str(&format!("{content}\n")),
                None => break,
            },
        }
    }
    Err(refuse(
        "script_metadata_invalid",
        "the `# /// script` block has no closing `# ///`",
    ))
}

#[cfg(unix)]
fn executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;
    fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn executable(_path: &Path) -> bool {
    false
}
