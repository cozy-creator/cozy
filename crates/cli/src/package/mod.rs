//! `cozy package`: find packages on a Hub, keep them on machines, and publish your own.
//!
//! Install, update and remove are runs on a machine through Calcifer; the machine resolves and
//! installs releases. Only `lock` and `publish` need uv, on the author's computer, and neither
//! runs package code.
use std::{path::PathBuf, process::ExitCode};

use clap::Subcommand;
use protocol::error::{Error, Kind};
use protocol::names::ROOT;

use crate::{Context, run::Placement};

mod author;
mod carried;
mod catalog;
pub(crate) mod live;
pub(crate) mod local;
mod machine;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Search the Hub's packages.
    Search {
        /// Words to match; none lists every package.
        query: Vec<String>,
        /// At most this many results.
        #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=200))]
        limit: u32,
    },
    /// A published package's releases, newest first; one release, with what its publish warned
    /// of (its owner sees the warnings).
    Info {
        /// [org/]name[@release]
        package: String,
    },
    /// Install a published package ([org/]name[@release]), a directory or a script on a machine:
    /// its environment, no weights. Without a release, the newest; over an installed package, its
    /// older releases there go (an update). Follows the install; Ctrl-C stops following, never it.
    Install {
        /// [org/]name[@release], a directory such as `.` or `./project`, or a PEP 723 `script.py`.
        #[arg(add = clap_complete::ArgValueCompleter::new(crate::completion::local))]
        package: String,
        #[command(flatten)]
        placement: Placement,
    },
    /// Remove packages from a machine, every Hub's installation unless --tensorhub narrows.
    Remove {
        /// [org/]name or local/name.
        #[arg(required = true)]
        packages: Vec<String>,
        #[command(flatten)]
        placement: Placement,
    },
    /// The packages your ready machines hold, live on a terminal.
    List {
        /// Only this machine.
        #[arg(long, value_name = "MACHINE")]
        on: Option<String>,
    },
    /// Lock this project's dependencies for the Hub and account it publishes to (needs uv).
    Lock {
        /// Let this dependency move to its newest compatible release; repeat for each.
        #[arg(long = "upgrade-package", value_name = "PACKAGE")]
        upgrade: Vec<String>,
        /// For the root namespace (first-party, Hub admins), not your account.
        #[arg(long)]
        root: bool,
    },
    /// Publish this project as <account>/<name>@<version>, or <name>@<version> with --root (needs uv).
    Publish {
        /// At the root namespace: first-party, Hub admins only.
        #[arg(long)]
        root: bool,
    },
    /// Yank a release: machines that hold it keep it, only an exact pin installs it.
    Yank {
        /// [org/]name@release
        release: String,
        /// Un-yank it instead: the unversioned default again (a gated candidate's promotion).
        #[arg(long)]
        undo: bool,
    },
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.command {
        Command::Search { query, limit } => catalog::search(ctx, &query.join(" "), limit).await,
        Command::Info { package } => catalog::info(ctx, &PackageRef::parse(&package)?).await,
        Command::Install { package, placement } => match directory(&package) {
            Some(path) => machine::install_directory(ctx, &path, placement.on()).await,
            None => machine::install(ctx, &PackageRef::parse(&package)?, placement.on()).await,
        },
        Command::Remove {
            packages,
            placement,
        } => machine::remove(ctx, &names(&packages)?, placement.on()).await,
        Command::List { on } => machine::list(ctx, on.map(Into::into)).await,
        Command::Lock { upgrade, root } => author::lock(ctx, &upgrade, root).await,
        Command::Publish { root } => author::publish(ctx, root).await,
        Command::Yank { release, undo } => {
            author::yank(ctx, &PackageRef::parse(&release)?, undo).await
        }
    }
}

/// `[org/]name[@release]`, lowercased: names are one identity in any case. No org is the root's
/// (first-party), org [`ROOT`] on the wire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackageRef {
    pub org: String,
    pub name: String,
    pub release: Option<String>,
}

impl PackageRef {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let text = text.trim();
        let (path, release) = match text.split_once('@') {
            Some((path, release)) => (path, Some(release.to_owned())),
            None => (text, None),
        };
        let valid = |part: &str| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        };
        match path.split_once('/').or(Some((ROOT, path))) {
            Some((org, name))
                if valid(org) && valid(name) && release.as_deref().is_none_or(valid) =>
            {
                Ok(Self {
                    org: org.to_ascii_lowercase(),
                    name: name.to_ascii_lowercase(),
                    release,
                })
            }
            _ => Err(Error::new(
                Kind::Usage,
                "package_ref_invalid",
                format!("{text:?} is not [org/]name[@release]"),
            )),
        }
    }

    pub fn package(&self) -> String {
        format!("{}/{}", self.org, self.name)
    }
}

impl std::fmt::Display for PackageRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.release {
            Some(release) => write!(f, "{}/{}@{release}", self.org, self.name),
            None => write!(f, "{}/{}", self.org, self.name),
        }
    }
}

/// Local code only when it is written as a path (`.`, `..`, absolute, `./…`, `../…`) or names a
/// `.py` script; a bare word is a package name.
fn directory(text: &str) -> Option<PathBuf> {
    let explicit = text.ends_with(".py")
        || text == "."
        || text == ".."
        || ["./", "../", ".\\", "..\\"]
            .iter()
            .any(|p| text.starts_with(p))
        || std::path::Path::new(text).is_absolute();
    explicit.then(|| PathBuf::from(text))
}

/// Names to remove: `[org/]name` (or `local/name`), with no release.
fn names(texts: &[String]) -> Result<Vec<String>, Error> {
    texts
        .iter()
        .map(|text| match PackageRef::parse(text)? {
            PackageRef {
                release: Some(_), ..
            } => Err(Error::new(
                Kind::Usage,
                "package_release_not_taken",
                format!("{text} names a release; name the package alone"),
            )),
            parsed => Ok(parsed.package()),
        })
        .collect()
}

/// A Hub timestamp in whole seconds: its nanoseconds are noise in a listing.
pub(crate) fn stamp(at: &str) -> String {
    match at.split_once('.') {
        Some((seconds, fraction)) => {
            let zone = fraction
                .find(|c: char| !c.is_ascii_digit())
                .map_or("", |at| &fraction[at..]);
            format!("{seconds}{zone}")
        }
        None => at.to_owned(),
    }
}

/// `items` newest first by TensorFS's one version ordering (`format::newest_by`); those whose
/// version does not parse last, as given.
pub(crate) fn newest_first<T>(mut items: Vec<T>, version: impl Fn(&T) -> &str) -> Vec<T> {
    let mut sorted = Vec::with_capacity(items.len());
    loop {
        let versions = items.iter().enumerate().map(|(i, item)| (i, version(item)));
        let Some((newest, _)) = tensorfs::format::newest_by(versions, |(_, v)| *v) else { break };
        sorted.push(items.remove(newest));
    }
    sorted.extend(items);
    sorted
}
