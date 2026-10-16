//! `cozy package`: find packages on a Hub, keep them on machines, and publish your own.
//!
//! Only `lock` and `publish` need uv, on the author's computer, and neither runs package code.
use std::process::ExitCode;

use clap::Subcommand;
use cozy_proto::error::{Error, Kind};

use crate::Context;

mod author;
mod catalog;
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
