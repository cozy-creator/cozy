//! `cozy package`: find packages on a Hub, keep them on machines, and publish your own.
//!
//! Install, update and remove are runs on a machine through Calcifer; the machine resolves and
//! installs releases. Only `lock` and `publish` need uv, on the author's computer, and neither
//! runs package code.
use std::{path::PathBuf, process::ExitCode};

use clap::Subcommand;
use cozy_proto::error::{Error, Kind};

use crate::{Context, run::Placement};

mod author;
mod catalog;
pub(crate) mod local;
mod machine;
mod registry;

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
    /// A published package's releases, newest first.
    Info {
        /// org/name
        package: String,
    },
    /// Install a published package (org/name[@release]), a directory or a script on a machine: its
    /// environment, no weights.
    Install {
        /// org/name[@release], a directory such as `.` or `./project`, or a PEP 723 `script.py`.
        package: String,
        #[command(flatten)]
        placement: Placement,
        /// Print the run and return instead of following it.
        #[arg(long)]
        detach: bool,
    },
    /// Move a machine's packages to their Hub's newest release: the named ones, else every one.
    Update {
        /// org/name; none updates every published package on the machine.
        packages: Vec<String>,
        #[command(flatten)]
        placement: Placement,
    },
    /// Remove packages from a machine, every Hub's installation unless --tensorhub narrows.
    Remove {
        /// org/name or local/name.
        #[arg(required = true)]
        packages: Vec<String>,
        #[command(flatten)]
        placement: Placement,
    },
    /// The packages your ready machines hold.
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
    },
    /// Publish this project as <account>/<name>@<version> (needs uv).
    Publish,
    /// Yank a release: machines that hold it keep it, nothing new installs it.
    Yank {
        /// org/name@release
        release: String,
    },
}

pub async fn exec(ctx: &Context, args: Args) -> Result<ExitCode, Error> {
    match args.command {
        Command::Search { query, limit } => catalog::search(ctx, &query.join(" "), limit).await,
        Command::Info { package } => catalog::info(ctx, &PackageRef::parse(&package)?).await,
        Command::Install {
            package,
            placement,
            detach,
        } => match directory(&package) {
            Some(path) => machine::install_directory(ctx, &path, placement.on(), detach).await,
            None => {
                machine::install(ctx, &PackageRef::parse(&package)?, placement.on(), detach).await
            }
        },
        Command::Update {
            packages,
            placement,
        } => machine::update(ctx, &names(&packages)?, placement.on()).await,
        Command::Remove {
            packages,
            placement,
        } => machine::remove(ctx, &names(&packages)?, placement.on()).await,
        Command::List { on } => machine::list(ctx, on.map(Into::into)).await,
        Command::Lock { upgrade } => author::lock(ctx, &upgrade).await,
        Command::Publish => author::publish(ctx).await,
        Command::Yank { release } => author::yank(ctx, &PackageRef::parse(&release)?).await,
    }
}

/// `org/name[@release]`, lowercased: names are one identity in any case.
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
        match path.split_once('/') {
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
                format!("{text:?} is not org/name or org/name@release"),
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

/// Names for update and remove: `org/name` (or `local/name`), with no release.
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

/// Releases in version order: finals after pre-releases of the same numbers, then by each number
/// (the order TensorD picks the newest by).
pub(crate) fn release_order(release: &str) -> (Vec<u64>, bool, Vec<u64>) {
    let numbers = |text: &str| -> Vec<u64> {
        text.split(|c: char| !c.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().unwrap_or(u64::MAX))
            .collect()
    };
    let (version, tag) = match release.find(|c: char| !(c.is_ascii_digit() || c == '.')) {
        Some(at) => (&release[..at], Some(&release[at..])),
        None => (release, None),
    };
    (
        numbers(version),
        tag.is_none(),
        tag.map(numbers).unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_are_lowercase_and_releases_order_by_number() {
        assert_eq!(
            PackageRef::parse("Fidika/SDXL").unwrap().package(),
            "fidika/sdxl"
        );
        assert!(PackageRef::parse("sdxl").is_err());
        assert!(directory("sdxl").is_none() && directory("./sdxl").is_some());
        let mut releases = ["2.9.0", "2.10.0", "2.10.0-rc.1", "1.37.2"];
        releases.sort_by_key(|r| release_order(r));
        assert_eq!(releases, ["1.37.2", "2.9.0", "2.10.0-rc.1", "2.10.0"]);
        assert_eq!(stamp("2026-10-16T09:34:25.879488Z"), "2026-10-16T09:34:25Z");
        assert_eq!(
            stamp("2026-10-16T09:34:25+02:00"),
            "2026-10-16T09:34:25+02:00"
        );
    }
}
