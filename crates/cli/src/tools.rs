//! `cozy tools`: the tools this computer has of those cozy uses and never bundles, and installing
//! the ones cozy fetches itself (pinned builds held to their sha256) into `~/.cozy/tools`. The
//! lookups features call (`ffmpeg`, `uv`) live here too.
use std::{
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::ExitCode,
};

use platform::tools::{self, Archive, Class, Found, Install, Side, TOOLS, Tool};
use protocol::error::{Error, Kind};
use futures_util::StreamExt as _;
use serde::Serialize;
use sha2::{Digest as _, Sha256};

use crate::{Context, output};

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Each tool cozy uses on this computer: found where, or what its absence costs.
    List,
    /// Fetch a tool's pinned build into ~/.cozy/tools, held to its sha256.
    Install {
        tool: Fetched,
        /// Do nothing when a usable one is already here.
        #[arg(long)]
        if_missing: bool,
    },
}

/// The tools cozy fetches itself.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Fetched {
    Ffmpeg,
    Uv,
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.command {
        Command::List => list(ctx),
        Command::Install { tool, if_missing } => {
            let (name, found) = match tool {
                Fetched::Ffmpeg => {
                    let found = (if_missing).then(|| usable_ffmpeg(ctx)).flatten();
                    let found = match found {
                        Some(found) => (found.ffmpeg, found.build, false),
                        None => {
                            let ffmpeg = install_ffmpeg(ctx).await?;
                            (ffmpeg.ffmpeg, ffmpeg.build, true)
                        }
                    };
                    ("ffmpeg", found)
                }
                Fetched::Uv => {
                    let found = if_missing
                        .then(|| tools::uv(&ctx.home.tools(), &path()))
                        .flatten();
                    let found = match found {
                        Some(found) => (found.path, found.version, false),
                        None => {
                            let uv = install_uv(ctx).await?;
                            (uv.path, uv.version, true)
                        }
                    };
                    ("uv", found)
                }
            };
            let (path, version, installed) = found;
            if ctx.json {
                output::json(&serde_json::json!({
                    "tool": name, "path": path, "version": version, "installed": installed,
                }));
            } else {
                output::Record::new()
                    .field(name, version)
                    .field("path", path.display().to_string())
                    .note(match installed {
                        true => "installed",
                        false => "already here: nothing changed",
                    })
                    .print(ctx);
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// One tool as `cozy tools list` and `cozy doctor` show it.
#[derive(Serialize)]
pub(crate) struct Row {
    pub name: &'static str,
    pub side: &'static str,
    pub class: &'static str,
    pub found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub version: String,
    pub cost: &'static str,
    /// What installs it here: a cozy verb, the OS's own command, or what to install.
    pub install: String,
}

/// Every tool this computer uses, probed now.
pub(crate) fn rows(ctx: &Context, side: Side) -> Vec<Row> {
    let path = path();
    TOOLS
        .iter()
        .filter(|tool| tool.side == side)
        .map(|tool| {
            let found: Option<Found> = (tool.probe)(&ctx.home.tools(), &path);
            Row {
                name: tool.name,
                side: match tool.side {
                    Side::Computer => "computer",
                    Side::Machine => "machine",
                },
                class: match tool.class {
                    Class::Essential => "essential",
                    Class::Optional => "optional",
                },
                found: found.is_some(),
                version: found
                    .as_ref()
                    .map(|f| f.version.clone())
                    .unwrap_or_default(),
                path: found.map(|f| f.path),
                cost: tool.cost,
                install: install_hint(tool, &path),
            }
        })
        .collect()
}

/// The words that install `tool` here.
fn install_hint(tool: &Tool, path: &[PathBuf]) -> String {
    match tool.install {
        Install::Cozy(verb) => verb.into(),
        Install::System(package) => tools::system_install(package, path)
            .unwrap_or_else(|| format!("install {package} with this computer's package manager")),
        Install::Note(text) => format!("install {text}"),
    }
}

/// A row as a line of `cozy doctor` and `cozy tools list`.
pub(crate) fn cells(row: &Row) -> [String; 3] {
    match row.found {
        true => [
            row.name.into(),
            match row.version.is_empty() {
                true => "found".into(),
                false => row.version.clone(),
            },
            row.path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        ],
        false => [
            row.name.into(),
            format!("missing: {}", row.cost),
            row.install.clone(),
        ],
    }
}

fn list(ctx: &Context) -> Result<ExitCode, Error> {
    let rows = rows(ctx, Side::Computer);
    if ctx.json {
        output::json(&serde_json::json!({ "tools": rows }));
        return Ok(ExitCode::SUCCESS);
    }
    let mut table = output::Table::new(ctx, ["TOOL", "STATE", "PATH OR INSTALL"]);
    for row in &rows {
        table = table.row(cells(row));
    }
    table.print();
    Ok(ExitCode::SUCCESS)
}

/// This process's PATH, as the probes take it.
fn path() -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default()
}

/// ffmpeg and ffprobe, with the build that answers (`ffmpeg -version`'s first line).
#[derive(Clone, Debug)]
pub struct Ffmpeg {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub build: String,
}

/// An ffmpeg that encodes libx264 here, the pinned build first.
fn usable_ffmpeg(ctx: &Context) -> Option<Ffmpeg> {
    let path = path();
    let (found, ffprobe) = tools::ffmpeg(&ctx.home.tools(), &path)?;
    tools::ffmpeg_encodes(&found.path, "libx264").then_some(Ffmpeg {
        ffmpeg: found.path,
        ffprobe,
        build: found.version,
    })
}

/// ffmpeg for a feature that needs it: the pinned build, else PATH's when it encodes libx264,
/// else the pinned build fetched now. With `--json` nothing is fetched: the refusal names the
/// command that installs it.
pub async fn ffmpeg(ctx: &Context) -> Result<Ffmpeg, Error> {
    if let Some(found) = usable_ffmpeg(ctx) {
        return Ok(found);
    }
    let path = path();
    let without = tools::ffmpeg(&ctx.home.tools(), &path).is_some();
    if ctx.json {
        let (code, why) = match without {
            true => (
                "ffmpeg_without_libx264",
                "this computer's ffmpeg cannot encode libx264",
            ),
            false => ("ffmpeg_unavailable", "this computer has no ffmpeg"),
        };
        return Err(Error::new(Kind::Unavailable, code, why).next("cozy tools install ffmpeg"));
    }
    install_ffmpeg(ctx).await
}

/// uv for authoring: the pinned build, else PATH's, else the pinned build fetched now (with
/// `--json` too: authoring cannot go on without it).
pub async fn uv(ctx: &Context) -> Result<PathBuf, Error> {
    if let Some(found) = tools::uv(&ctx.home.tools(), &path()) {
        return Ok(found.path);
    }
    install_uv(ctx).await.map(|found| found.path)
}

async fn install_ffmpeg(ctx: &Context) -> Result<Ffmpeg, Error> {
    let archives = tools::ffmpeg_build();
    if archives.is_empty() {
        return Err(unpinned(ctx, "ffmpeg"));
    }
    let dir = tools::ffmpeg_dir(&ctx.home.tools());
    fetch(ctx, "ffmpeg", archives, &dir).await?;
    usable_ffmpeg(ctx).ok_or_else(|| {
        Error::new(
            Kind::Internal,
            "ffmpeg_unavailable",
            format!("the pinned ffmpeg in {} does not answer", dir.display()),
        )
    })
}

async fn install_uv(ctx: &Context) -> Result<Found, Error> {
    let Some(archive) = tools::uv_build() else {
        return Err(unpinned(ctx, "uv"));
    };
    let dir = tools::uv_dir(&ctx.home.tools());
    fetch(ctx, "uv", std::slice::from_ref(&archive), &dir).await?;
    tools::uv(&ctx.home.tools(), &[]).ok_or_else(|| {
        Error::new(
            Kind::Internal,
            "uv_unavailable",
            format!("the pinned uv in {} does not answer", dir.display()),
        )
    })
}

/// The refusal a feature gives when an essential tool is missing: what its absence costs, what
/// installs it, and `cozy doctor`.
pub fn missing(ctx: &Context, name: &str) -> Error {
    let tool = tools::tool(name);
    Error::new(
        Kind::Unavailable,
        format!("{name}_unavailable"),
        format!("{name} not found: {}", tool.cost),
    )
    .next(install_hint(tool, &path()))
    .next(doctor(ctx))
}

fn doctor(ctx: &Context) -> &'static str {
    match ctx.json {
        true => "cozy doctor --json",
        false => "cozy doctor",
    }
}

/// No build is pinned for this OS/CPU: the OS's own command installs it.
fn unpinned(ctx: &Context, name: &'static str) -> Error {
    let tool = tools::tool(name);
    Error::new(
        Kind::Unavailable,
        format!("{name}_unavailable"),
        format!(
            "cozy pins no {name} build for {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
    )
    .next(install_hint(
        &Tool {
            install: Install::System(name),
            ..*tool
        },
        &path(),
    ))
    .next(doctor(ctx))
}

/// Downloads `archives`, each held to its sha256, and puts their members into `dir` whole: a
/// staging directory renamed into place, so `dir` is either complete or absent.
async fn fetch(ctx: &Context, name: &str, archives: &[Archive], dir: &Path) -> Result<(), Error> {
    let failed = |e: String| {
        Error::new(Kind::Unavailable, format!("{name}_unavailable"), e)
            .next(format!("cozy tools install {name}"))
            .next(doctor(ctx))
    };
    let parent = dir.parent().expect("a tool's directory has a parent");
    std::fs::create_dir_all(parent).map_err(|e| failed(format!("{}: {e}", parent.display())))?;
    let staging = tempfile::tempdir_in(parent).map_err(|e| failed(e.to_string()))?;
    for archive in archives {
        output::say(ctx, &format!("fetching {name}: {}", archive.url));
        let mut download = tempfile::tempfile_in(parent).map_err(|e| failed(e.to_string()))?;
        let response = ctx
            .http()
            .get(archive.url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| failed(format!("{}: {e}", archive.url)))?;
        let mut hash = Sha256::new();
        let mut body = response.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|e| failed(format!("{}: {e}", archive.url)))?;
            hash.update(&chunk);
            download
                .write_all(&chunk)
                .map_err(|e| failed(e.to_string()))?;
        }
        let got = hex::encode(hash.finalize());
        if got != archive.sha256 {
            return Err(Error::new(
                Kind::Validation,
                format!("{name}_digest_mismatch"),
                format!(
                    "{} is sha256:{got}, not the pinned sha256:{}",
                    archive.url, archive.sha256
                ),
            ));
        }
        extract(archive, download, staging.path()).map_err(|e| failed(e.to_string()))?;
    }
    let staged = staging.keep();
    match std::fs::rename(&staged, dir) {
        Ok(()) => Ok(()),
        // Another command installed it meanwhile: theirs is as good.
        Err(_) if dir.is_dir() => {
            let _ = std::fs::remove_dir_all(&staged);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staged);
            Err(failed(format!("{}: {e}", dir.display())))
        }
    }
}

/// `archive`'s members into `dir`, runnable.
fn extract(archive: &Archive, mut file: std::fs::File, dir: &Path) -> io::Result<()> {
    use std::io::Seek as _;
    file.rewind()?;
    for (member, name) in archive.members {
        let out = dir.join(name);
        let mut target = std::fs::File::create(&out)?;
        if archive.url.ends_with(".zip") {
            let mut zip = zip::ZipArchive::new(&mut file).map_err(io::Error::other)?;
            let mut entry = zip.by_name(member).map_err(io::Error::other)?;
            io::copy(&mut entry, &mut target)?;
        } else {
            untar(&mut file, member, &mut target)?;
        }
        file.rewind()?;
        runnable(&out)?;
    }
    Ok(())
}

#[cfg(unix)]
fn untar(file: &mut std::fs::File, member: &str, out: &mut std::fs::File) -> io::Result<()> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.path()?.as_os_str() == member {
            io::copy(&mut entry, out)?;
            return Ok(());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("the archive holds no {member}"),
    ))
}

/// Windows archives are zips.
#[cfg(not(unix))]
fn untar(_: &mut std::fs::File, member: &str, _: &mut std::fs::File) -> io::Result<()> {
    Err(io::Error::other(format!(
        "{member}: a tar archive on Windows"
    )))
}

fn runnable(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
    }
    let _ = path;
    Ok(())
}
